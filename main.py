from pathlib import Path
from tqdm import tqdm
from torchcrf import CRF     
import yaml, os, torch, torchvision, torch.nn as nn
from torchvision.io import read_video
from torchvision.models import mobilenet_v2
from data_process import data_load, audio_mfcc, boundary_to_tags

with open("cfg.yaml", 'r') as f:
    data = yaml.safe_load(f)
    batch_size = data["batch_size"]
    learn_rate = float(data["learn_rate"])
    fps, resolution = data["fps"], data["resolution"]

class IntroDetector(nn.Module):
    """Labels: 0 = OTHER, 1 = INTRO, 2 = RECAP (optional)"""
    def __init__(self, cnn_out=512, audio_dim=64, hidden=256, n_labels=2):
        super().__init__()
        # self.cnn = torch.hub.load("pytorch/vision", "resnet50", weights="ResNet50_Weights.DEFAULT")
        self.cnn = mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        # self.cnn = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.cnn.fc = nn.Identity()
        for p in self.cnn.parameters():
            p.requires_grad = False

        self.proj_audio = nn.Linear(audio_dim, 128)
        self.lstm = nn.LSTM(cnn_out+128, hidden, 2, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden*2, n_labels)
        self.crf = CRF(n_labels, batch_first=True)

    def forward(self, frames, audio_feats, tags=None, mask=None):
        # frames: [B,T,3,H,W]  audio_feats: [B,T,64]
        B,T,_,H,W = frames.shape
        v = self.cnn(frames.flatten(0,1))
        v = v.view(B,T,-1)
        a = self.proj_audio(audio_feats)
        x = torch.cat([v, a], dim=-1)
        h,_ = self.lstm(x)
        logits = self.classifier(self.drop(h))
        if tags is None:
            return self.crf.decode(logits, mask)
        else:
            loss = -self.crf(logits, tags, mask, reduction="mean")
            return loss


model = IntroDetector(cnn_out=128, audio_dim=64, hidden=64, n_labels=2).to("cuda")
optim = torch.optim.SGD(model.parameters(), lr=learn_rate)

for epoch in range(5):
    for frames, audio, tags in data_load(batch_size, resolution, fps):
        # video = videos[0]
        # vid_path = Path("data_processed") / f"{video}.mp4"

        # frames= read_frames(vid_path, fps=fps).to("cuda")

        # tags = tags[0].to("cuda")

        frames = frames.to("cuda")
        audio = audio.to("cuda")
        tags = tags.to("cuda")

        mask = torch.ones_like(tags, dtype=torch.bool)

        loss = model(
            frames.unsqueeze(0),
            audio.unsqueeze(0),
            tags.unsqueeze(0),
            mask.unsqueeze(0),
        )
        loss.backward()
        optim.step()
        optim.zero_grad()

        # print(loss.item())

