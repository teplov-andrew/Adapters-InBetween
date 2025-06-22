from pathlib import Path
from base import BaseAdapter

class TpsAdapter(BaseAdapter):
    def __init__(self,
                 xN=30, save_path="./output",
                 cpu=False, model_path="tps-inbetween/ckpt/model_latest.pt"):
        self.xN, self.save_path, self.cpu, self.model_path = xN, save_path, cpu, model_path

    def __call__(self, img1, img2, **kw):
        cmd = [
            "python", "tps-inbetween/demo.py",
            "--image1", str(Path(img1)),
            "--image2", str(Path(img2)),
            "--xN", str(self.xN),
            "--save_path", self.save_path,
            "--model_path", self.model_path,
        ]
        if self.cpu:
            cmd.append("--cpu")
        return self._run(cmd, **kw)