# sain.py
from pathlib import Path
from base import BaseAdapter

class SainAdapter(BaseAdapter):
    def __init__(self,
                 ckpt="SAIN/checkpoint/model_best.pth",
                 size=512,
                 device="cuda"):
        if size % 8:
            raise ValueError("--size must be multiple of 8")
        self.ckpt, self.size, self.device = ckpt, size, device.lower()

    def __call__(self, img1, img2, **kw):
        out_dir = "output"          # ./output
        stemA, stemB = Path(img1).stem, Path(img2).stem
        out_file = f"{out_dir}/result_{stemA}_{stemB}_sain.png"

        cmd = [
            "python", "SAIN/infer.py",
            "--frameA", str(Path(img1)),
            "--frameB", str(Path(img2)),
            "--out",    str(out_file),
            "--ckpt",   self.ckpt,
            "--size",   str(self.size),
            "--device", self.device,
        ]
        return self._run(cmd, **kw)
