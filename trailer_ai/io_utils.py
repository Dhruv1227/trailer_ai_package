import os, subprocess
def run(cmd: list, check: bool = True):
    print(">>", " ".join([str(c) for c in cmd]))
    return subprocess.run(cmd, check=check)
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
def basename_noext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]
