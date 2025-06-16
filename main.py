import os
from torchcrf import CRF      
import torch, torch.nn as nn
from torchvision.io import read_video
from torch.utils.data import DataLoader, Dataset, TensorDataset
from data_process import *


class IntroDetector(nn.Module):
    """
    Visual CNN (frozen) + audio MFCC → concat → Bi-LSTM → Drop → Linear → CRF
    Labels: 0 = OTHER, 1 = INTRO, 2 = RECAP (optional)
    """
    def __init__(self, cnn_out=512, audio_dim=64, hidden=256, n_labels=2):
        super().__init__()
        self.cnn = torch.hub.load("pytorch/vision", "resnet50", pretrained=True) # ResNet-50 / VideoMAE, Swin, etc.
        self.cnn.fc = nn.Identity()
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.proj_audio = nn.Linear(audio_dim, 128)   # make dims similar
        self.lstm = nn.LSTM(cnn_out+128, hidden, 2, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden*2, n_labels)
        self.crf = CRF(n_labels, batch_first=True)

    def forward(self, frames, audio_feats, tags=None, mask=None):
        # frames: [B,T,3,H,W]  audio_feats: [B,T,64]
        B,T,_,H,W = frames.shape
        v = self.cnn(frames.flatten(0,1))          # [B*T,512]
        v = v.view(B,T,-1)                         # [B,T,512]
        a = self.proj_audio(audio_feats)           # [B,T,128]
        x = torch.cat([v, a], dim=-1)              # [B,T,640]
        h,_ = self.lstm(x)
        logits = self.classifier(self.drop(h))     # [B,T,n_labels]
        if tags is None:
            return self.crf.decode(logits, mask)
        else:
            loss = -self.crf(logits, tags, mask, reduction="mean")
            return loss

model = IntroDetector()
optim = torch.optim.SGD(lr=0.001)
os.chdir("data/data_train_short")
for i in os.curdir():
    


loader = DataLoader()

for epoch in range(20):
    for video, start, stop in loader:
        frames   = read_frames(video, fps=2).to("cuda")        # [T,3,H,W]
        audio    =  audio_mfcc(video, fps=2).to("cuda")        # [T,64]
        tags     = boundary_to_tags(start*2, stop*2, frames.shape[0]).to("cuda")
        mask     = torch.ones_like(tags, dtype=torch.bool)
        loss = model(frames.unsqueeze(0), audio.unsqueeze(0), tags.unsqueeze(0), mask.unsqueeze(0))
        loss.backward()
        optim.step() 
        optim.zero_grad()
