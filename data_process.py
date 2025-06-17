import json, yaml
import shutil
import torch, torchaudio, torch.nn as nn
from torchvision.io import read_video
from datetime import timedelta
from pathlib import Path

import torch, cv2
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as dF
import nvidia.dali.types as types


def as_seconds(x):
    if isinstance(x, (int, float)):
        return float(x)
    parts = list(map(int, x.split(":")))
    if len(parts) == 1:
        h, m, s = 0, 0, parts[0]
    elif len(parts) == 2:
        h, m, s = 0, *parts
    elif len(parts) == 3:
        h, m, s = parts
    return timedelta(hours=h, minutes=m, seconds=s).total_seconds()

def slice_video(src, start="00:00", end="15:00", dst="output", fps=10, resolution=(1920, 1080), codecs="libx264"):
    dst = Path("data_processed/" + dst + ".mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    input_stream = ffmpeg.input(src, ss=start, to=end)
    res = "x".join([str(i) for i in resolution]) 
    output_kwargs = {'vcodec': codecs, 'r': fps, 's': res}
    output_stream = ffmpeg.output(input_stream, dst.as_posix(), **output_kwargs)
    ffmpeg.run(output_stream, overwrite_output=True)

def read_frames(video_path, fps=2):
    vid, _, info = read_video(video_path, pts_unit="sec", output_format="TCHW")
    if fps >= info["video_fps"]:
        return vid
    step = int(info["video_fps"] // fps)
    return vid[::step]

class _VideoPipe(Pipeline):
    def __init__(self, file, step, resize_hw, batch_size=1, device_id=0):
        super().__init__(batch_size=batch_size, num_threads=2, device_id=device_id)
        self.reader = dF.readers.video(
            device      = "gpu",
            filenames   = [file],
            sequence_length = 0,      # full clip
            step        = step,
            dtype       = types.UINT8,
            random_shuffle = False,
            normalized  = False)
        self.resize_hw = resize_hw

    def define_graph(self):
        seq = self.reader()           # [B,T,H,W,3] on GPU
        if self.resize_hw:
            seq = dF.resize(seq, size=self.resize_hw)
        return seq

def read_frames_dali(video_path, fps=2, resize_hw=None, device_id=0):
    cap = cv2.VideoCapture(str(video_path))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    step = max(1, round(in_fps / fps))
    pipe = _VideoPipe(str(video_path), step, resize_hw, device_id=device_id)
    pipe.build()
    out, = pipe.run() 
    vid  = out.as_cpu().as_array()[0]
    return (torch.from_numpy(vid).permute(0,3,1,2).float().div_(255))    

def audio_mfcc(video_path, fps=1, sample_rate=16000):
    print(1)
    _mfcc = torchaudio.transforms.MFCC(sample_rate=16000,n_mfcc=64,melkwargs={'n_mels': 64,'n_fft': 1024,'hop_length': 512})
    data, sr = torchaudio.load(video_path, format="wav", normalize=True)
    if sr != sample_rate:
        data = torchaudio.functional.resample(data, sr, sample_rate)
    L = data.shape[-1] // sample_rate
    data = data[:, :L*sample_rate].reshape(1, L, sample_rate)
    feats = _mfcc(data.flatten(0,1))
    print(2)
    return feats.mean(-1)

def boundary_to_tags(start_s, stop_s, total_frames, fps):
    tags = torch.zeros(total_frames, dtype=torch.long)
    start_idx = int(round(start_s * fps))
    stop_idx  = int(round(stop_s  * fps))
    tags[start_idx:stop_idx] = 1
    return tags
    
def data_load(batch_size, resolution, fps, dali = False):
    with open("data_processed/train_labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    videos,audios,stamps = [],[],[]
    for i, (name, info) in enumerate(labels.items()):
        
        # vid_path = Path("data/data_train_short") / name / f"{name}.mp4"
        vid_path = "data_processed/" + name + ".mp4"
        if dali:
            videos.append(read_frames_dali(vid_path, fps=fps))
        else:
            videos.append(read_frames(vid_path, fps=fps))
        stamps.append(boundary_to_tags(info["start"], info["stop"], videos[-1].shape[0], fps))
        audios.append(audio_mfcc(vid_path, fps=fps).to("cuda"))
        if (i + 1) % batch_size == 0 or i == len(labels) - 1:
            yield torch.cat(videos, dim=0), torch.cat(audios, dim=0), torch.cat(stamps, dim=0)
            videos,audios,stamps = [],[],[]

if __name__ == "__main__":

    with open("cfg.yaml", 'r') as f:
        data = yaml.safe_load(f)
        train_len = data["train_len"]
        fps, resolution = data["fps"], data["resolution"]
    with open("data/labels_json/train_labels.json","r", encoding="utf-8") as f:
        train_labels = json.load(f)
    with open("data/labels_json/test_labels.json", "r", encoding="utf-8") as f:
        test_labels = json.load(f)

    dir = Path("data_processed")
    if dir.exists() and dir.is_dir():
        shutil.rmtree(dir)
    else:
        dir.mkdir(parents=True, exist_ok=True)

    videos = {}
    for i, (name, info) in enumerate(train_labels.items()):
        st, en = as_seconds(info["start"]), as_seconds(info["end"])
        
        video = Path("data/data_train_short") / name / f"{name}.mp4"
        slice_video(video, 0, en * 2, name, fps, resolution)
        videos[name] = {
            "name": name,
            "start": st, 
            "stop": en, 
            "fps": fps,
            "resolutionx":resolution[0],
            "resolutiony":resolution[1]
        }
        if i == train_len - 1:
            with open("data_processed/train_labels.json", "w", encoding="utf-8") as f:
                json.dump(videos, f, indent=4)
            break

    print("Data preprocessing completed.")
