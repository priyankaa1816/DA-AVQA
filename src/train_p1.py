"""
P1 - your architecture (reference). Training/evaluation protocol: trainer.py (shared).

Per segment k = 1..K (K = 4):
  FAST feat 768 -> Linear 128 -> LN
  CLIP 512 (mean of frames) -> Linear 64 -> LN
  Motion 4x32 -> TCN(d=1,2,4) -> mean -> Linear 64 -> LN
  Anim 20 -> MLP 32 -> 64 -> LN                        -> concat F_c (320)
  ELEMENT-WISE gating: g = sigmoid(MLP 320-64-320) ; F_g = F_c * g
  F_f = F_g + MLP(F_g)                 (320-128-320, dropout 0.1)
  alpha = softmax_k(Linear 320-64 -> tanh -> Linear 1) ; F_q = sum_k alpha_k F_f,k
  dQ = MLP(F_q) (320-64-1, dropout 0.1, last layer zero-init)
  MOS_hat = Q_FAST_calibrated + dQ

Usage:
  python train_p1.py --work_dir D:/avqa_work --mode cv
"""
import torch
import torch.nn as nn

from blocks import BRANCH_DIMS, BranchEncoders, FusionAttentionHead, gate_branch_summary
from trainer import build_parser, run


class P1Model(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.enc = BranchEncoders()
        d = sum(BRANCH_DIMS)
        self.gate = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, d), nn.Sigmoid())
        self.post = FusionAttentionHead(d, dropout)

    def forward(self, fast, clip, motion, anim, q_fast):
        f_c = torch.cat(self.enc(fast, clip, motion, anim), -1)          # (B,K,320)
        g = self.gate(f_c)                                               # element-wise gate in [0,1]
        dq, alpha = self.post(f_c * g)
        return q_fast + dq, {"delta": dq, "alpha": alpha, "branch": gate_branch_summary(g)}


def main():
    args = build_parser(default_tag="P1").parse_args()
    run(args, lambda: P1Model(dropout=args.dropout), pipeline="P1")


if __name__ == "__main__":
    main()
