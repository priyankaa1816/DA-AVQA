"""
Shared model building blocks used by every pipeline (P1, P2, P3, P3b, P4).

Pipelines import ONLY from here (and trainer.py), never from each other, so each
train_pX.py works on its own as long as blocks.py / trainer.py / common.py are present.
"""
import torch
import torch.nn as nn

BRANCH_DIMS = [128, 64, 64, 64]          # fast, clip, motion, anim  -> 320 per segment
D_TEMPORAL = 128                         # width of the temporal encoder in P3 / P3b
CODE_VERSION = "2026-10-25"


class BranchEncoders(nn.Module):
    """Identical in every pipeline: 4 branches -> (B,K,128), (B,K,64), (B,K,64), (B,K,64)."""

    def __init__(self, d_fast=768, d_clip=512, n_motion=4, n_anim=20):
        super().__init__()
        self.fast_proj = nn.Sequential(nn.Linear(d_fast, 128), nn.LayerNorm(128))
        self.clip_proj = nn.Sequential(nn.Linear(d_clip, 64), nn.LayerNorm(64))
        self.tcn = nn.Sequential(
            nn.Conv1d(n_motion, 32, 3, padding=1, dilation=1), nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=4, dilation=4), nn.GELU(),
        )
        self.motion_proj = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64))
        self.anim_mlp = nn.Sequential(nn.Linear(n_anim, 32), nn.GELU(), nn.Linear(32, 64), nn.LayerNorm(64))

    def forward(self, fast, clip, motion, anim):
        B, K = fast.shape[:2]
        fa = self.fast_proj(fast)
        fb = self.clip_proj(clip)
        m = self.tcn(motion.reshape(B * K, motion.shape[2], motion.shape[3])).mean(-1)
        fc = self.motion_proj(m).view(B, K, -1)
        fd = self.anim_mlp(anim)
        return fa, fb, fc, fd


class FusionAttentionHead(nn.Module):
    """P1/P2: residual fusion MLP -> segment attention -> dQ head (last layer zero-init)."""

    def __init__(self, d=320, dropout=0.1):
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, d))
        self.att = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)     # start exactly at the calibrated FAST-VQA score
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, f_g):
        f_f = f_g + self.fusion(f_g)
        alpha = torch.softmax(self.att(f_f).squeeze(-1), dim=1)  # (B,K)
        f_q = (alpha.unsqueeze(-1) * f_f).sum(1)                 # (B,320)
        return self.head(f_q).squeeze(-1), alpha


class GatedFusion(nn.Module):
    """P3/P3b: P1's element-wise gate + residual fusion MLP (unchanged)."""

    def __init__(self, d=320, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, d), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, d))

    def forward(self, f_c):
        g = self.gate(f_c)
        f_g = f_c * g
        return f_g + self.fusion(f_g), g


class PoolHead(nn.Module):
    """P3/P3b: attention pooling over segments + dQ head (last layer zero-init)."""

    def __init__(self, d=D_TEMPORAL, dropout=0.1):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, h):
        alpha = torch.softmax(self.att(h).squeeze(-1), dim=1)   # (B,K)
        pooled = (alpha.unsqueeze(-1) * h).sum(1)
        return self.head(pooled).squeeze(-1), alpha


def gate_branch_summary(g):
    """Mean element-wise gate value of each branch's dimensions -> (B,K,4)."""
    return torch.stack([s.mean(-1) for s in torch.split(g, BRANCH_DIMS, dim=-1)], -1)
