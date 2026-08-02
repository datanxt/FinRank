#!/usr/bin/env python3
"""Run the full FinRank baseline harness on Modal (GPU).

Ships FinRank.jsonl + run_baselines.py into a GPU container, runs everything
there (dense encoding + cross-encoding on CUDA, sparse + metrics on CPU),
and writes results.json / ablations.json back to baselines/results/.

Run:  modal run modal_run.py
"""
import json, pathlib, modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.1", "scikit-learn==1.9.0", "rank-bm25==0.2.2",
                 "sentence-transformers==5.6.1", "transformers==5.14.1", "torch==2.13.0")
)
app = modal.App("finrank-baselines", image=image)


@app.function(gpu="T4", cpu=4.0, memory=16384, timeout=3600)
def run(jsonl_bytes: bytes, harness_src: str) -> dict:
    import subprocess, pathlib
    pathlib.Path("/tmp/data.jsonl").write_bytes(jsonl_bytes)
    pathlib.Path("/tmp/run_baselines.py").write_text(harness_src)
    proc = subprocess.run(
        ["python", "-u", "/tmp/run_baselines.py", "--data", "/tmp/data.jsonl", "--out", "/tmp/results"],
        capture_output=True, text=True,
    )
    out = {"log": proc.stdout[-8000:] + "\n--- stderr ---\n" + proc.stderr[-4000:], "rc": proc.returncode}
    for name in ("results.json", "ablations.json"):
        p = pathlib.Path("/tmp/results") / name
        if p.exists():
            out[name] = p.read_text()
    return out


@app.local_entrypoint()
def main():
    here = pathlib.Path(__file__).parent
    data_path = here.parent / "FinRank_normalized.jsonl"
    if not data_path.exists():
        data_path = here.parent / "FinRank.jsonl"
    data = data_path.read_bytes()
    src = (here / "run_baselines.py").read_text()
    print(f"sending {len(data)/1e6:.1f} MB to Modal (T4)...")
    out = run.remote(data, src)
    print(out["log"])
    if out["rc"] != 0:
        raise SystemExit(f"remote harness failed rc={out['rc']}")
    (here / "results").mkdir(exist_ok=True)
    for name in ("results.json", "ablations.json"):
        (here / "results" / name).write_text(out[name])
        print("wrote", here / "results" / name)
