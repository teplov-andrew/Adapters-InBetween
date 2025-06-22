from pathlib import Path
from base import BaseAdapter

class RifeAdapter(BaseAdapter):
    def __init__(self,
                 exp=4, ratio=0.0, rthreshold=.02, rmaxcyc=8,
                 model="ECCV2022-RIFE/train_log", out_dir="./output"):
        self.exp, self.ratio, self.rthreshold = exp, ratio, rthreshold
        self.rmaxcyc, self.model, self.out_dir = rmaxcyc, model, out_dir

    def __call__(self, img1, img2, **kw):
        cmd = [
            "python3", "ECCV2022-RIFE/inference_img.py",
            "--img", str(Path(img1)), str(Path(img2)),
            "--exp", str(self.exp),
            "--ratio", str(self.ratio),
            "--rthreshold", str(self.rthreshold),
            "--rmaxcycles", str(self.rmaxcyc),
            "--model", self.model,
        ]
        return self._run(cmd, **kw)