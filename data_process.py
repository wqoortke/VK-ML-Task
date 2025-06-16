from pathlib import Path
from torchcrf import CRF      
import torch, torchaudio, torch.nn as nn
from torchvision.io import read_video
from torch.utils.data import DataLoader, Dataset, TensorDataset
import ffmpeg

def slice_video(src, start="00:00", end="15:00", dst="file.mp4", fps=10, resolution=(1920, 1080), codecs="libx264"):
    dst = Path("data_processed/" + dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    input_stream = ffmpeg.input(src, ss=start, to=end)
    if isinstance(resolution, tuple): 
        res = "x".join(resolution) 
    output_kwargs = {'vcodec': codecs, 'r': fps, 's': res}
    output_stream = ffmpeg.output(input_stream, dst.as_posix(), **output_kwargs)
    ffmpeg.run(output_stream, overwrite_output=True)

def read_frames(video_path, fps=10):
    vid, _, info = read_video(video_path, pts_unit="sec", output_format="TCHW")
    native_fps = info["video_fps"]
    if fps >= native_fps:
        return vid
    step = int(native_fps // fps)
    return vid[::step]

_mfcc = torchaudio.transforms.MFCC(sample_rate=16000,n_mfcc=64,melkwargs={'n_mels': 64,'n_fft': 1024,'hop_length': 512})
def audio_mfcc(video_path, fps=1, sample_rate=16000):
    data, sr = torchaudio.load(video_path, format="wav", normalize=True)
    if sr != sample_rate:
        data = torchaudio.functional.resample(data, sr, sample_rate)
    L = data.shape[-1] // sample_rate
    data = data[:, :L*sample_rate].reshape(1, L, sample_rate)
    feats = _mfcc(data.flatten(0,1))               # [L, 64, time]
    return feats.mean(-1)                          # [L, 64]

def boundary_to_tags(start_s, stop_s, total_s):
    tags = torch.zeros(total_s, dtype=torch.long)
    tags[start_s:stop_s] = 1      # INTRO (0 = OTHER elsewhere)
    return tags
