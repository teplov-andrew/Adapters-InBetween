from rife import RifeAdapter
from tps  import TpsAdapter
from sain import SainAdapter


ADAPTERS = {
    "rife": RifeAdapter(),    
    "tps" : TpsAdapter(),
    "sain": SainAdapter(),
}

def run_all(img1: str, img2: str, *, verbose=True, **subproc_kw):
    results = {}
    for name, adapter in ADAPTERS.items():
        if verbose:
            print(f"\n=== {name.upper()} ===")
        results[name] = adapter(img1, img2, **subproc_kw)
    return results

imgA = "frame1.png"
imgB = "frame2.png"

logs = run_all(imgA, imgB, capture_output=True, text=True)

for name, proc in logs.items():
    print(f"\n--- {name} stdout ---")
    print(proc.stdout)