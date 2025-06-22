#!/usr/bin/env python3
"""
infer_fullflow.py — SAIN-хуки (points=1 + двунаправленный RAFT-flow)

Запуск:
 python infer_fullflow.py \
   --frameA frame1.png --frameB frame2.png \
   --out result.png --size 512 --device cuda
"""

import argparse, sys, torch
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from model.SAIN import SAIN

def pil2ten(img):
    return to_tensor(img).unsqueeze(0).sub_(0.5).div_(0.5)

def ten2pil(t):
    if t.ndim == 4: t = t[0]
    t = t.add(1).div(2).clamp(0,1)
    arr = (t.permute(1,2,0).cpu().numpy()*255).astype('uint8')
    return Image.fromarray(arr)

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frameA", required=True)
    p.add_argument("--frameB", required=True)
    p.add_argument("--out",    default="inbetween.png")
    p.add_argument("--ckpt",   default="checkpoint/model_best.pth")
    p.add_argument("--size",   type=int, required=True, help="кратна 8")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = torch.device(args.device)

    # 1) load & resize
    A_img = Image.open(args.frameA).convert("RGB").resize((args.size,)*2)
    B_img = Image.open(args.frameB).convert("RGB").resize((args.size,)*2)
    H = W = args.size

    # 2) tensors −1…1
    A = pil2ten(A_img).to(dev)
    B = pil2ten(B_img).to(dev)

    # 3) POINTS = ВСЁ ЕДИНИЦЫ
    points = torch.ones((1,1,H,W), device=dev)

    # 4) двунаправленный RAFT-flow
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(dev).eval()
    inpA = to_tensor(A_img).unsqueeze(0).to(dev)
    inpB = to_tensor(B_img).unsqueeze(0).to(dev)
    flowAB = raft(inpA, inpB)[-1]
    flowBA = raft(inpB, inpA)[-1]
    region_flow = [flowAB, flowBA]

    # 5) init SAIN с теми же гиперпараметрами
    sargs = type("",(),{})()
    sargs.device       = dev
    sargs.phase        = "test"
    sargs.crop_size    = H
    sargs.joinType     = "concat"
    sargs.c            = 64
    sargs.window_size  = 24
    sargs.resume_flownet = ""
    net = SAIN(sargs).to(dev).eval()

    # 6) load weights (игнорируем только attn_mask)
    ckpt = torch.load(args.ckpt, map_location=dev)
    state = ckpt.get("state_dict", ckpt)
    # strip prefixes
    newst = {}
    for k,v in state.items():
        k2 = k
        for pref in ("module.","sain."):
            if k2.startswith(pref): k2 = k2[len(pref):]
        if "attn_mask" in k2: continue
        newst[k2] = v
    net.load_state_dict(newst, strict=False)

    # 7) forward & save
    pred = net(A, B, points, region_flow)[0]
    ten2pil(pred).save(args.out)
    print("✓ saved →", args.out)

if __name__=="__main__":
    main()
