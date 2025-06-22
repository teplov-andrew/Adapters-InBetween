from abc import ABC, abstractmethod
import subprocess, shlex
from pathlib import Path
from typing import Any

class BaseAdapter(ABC):
    @abstractmethod
    def __call__(self, img1, img2, **run_kw): ...
        
    def _run(self, cmd: list[str], **kw):
        print("Executing:", shlex.join(cmd))
        return subprocess.run(cmd, check=True, **kw)