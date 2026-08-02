#!/usr/bin/env python3
"""Run run_extra_retrievers.py on Modal (A10G GPU).

Run:  modal run modal_extra.py
"""
import pathlib, modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.1", "scikit-learn==1.9.0", "rank-bm25==0.2.2",
                 "sentence-transformers==5.6.1", "transformers==5.14.1", "torch==2.13.0")
)
app = modal.App("finrank-extra-retrievers", image=image)


@app.function(gpu="A10G", cpu=4.0, memory=32768, timeout=3600,
              env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
def run(jsonl_bytes: bytes, harness_src: str, extra_src: str, prior_json: str = "") -> dict:
    import subprocess, pathlib
    pathlib.Path("/tmp/data.jsonl").write_bytes(jsonl_bytes)
    pathlib.Path("/tmp/run_baselines.py").write_text(harness_src)
    pathlib.Path("/tmp/run_extra_retrievers.py").write_text(extra_src)
    if prior_json:
        pathlib.Path("/tmp/results").mkdir(exist_ok=True)
        pathlib.Path("/tmp/results/extra_retrievers.json").write_text(prior_json)
    proc = subprocess.run(
        ["python", "-u", "/tmp/run_extra_retrievers.py", "--data", "/tmp/data.jsonl",
         "--out", "/tmp/results"],
        capture_output=True, text=True, cwd="/tmp",
    )
    out = {"log": proc.stdout[-8000:] + "\n--- stderr ---\n" + proc.stderr[-4000:], "rc": proc.returncode}
    p = pathlib.Path("/tmp/results/extra_retrievers.json")
    if p.exists():
        out["extra_retrievers.json"] = p.read_text()
    return out


@app.local_entrypoint()
def main():
    here = pathlib.Path(__file__).parent
    data = here.parent / "FinRank_normalized.jsonl"
    if not data.exists():
        data = here.parent / "FinRank.jsonl"
    out = run.remote(data.read_bytes(),
                     (here / "run_baselines.py").read_text(),
                     (here / "run_extra_retrievers.py").read_text())
    print(out["log"])
    if out["rc"] != 0:
        raise SystemExit(f"remote failed rc={out['rc']}")
    (here / "results").mkdir(exist_ok=True)
    (here / "results" / "extra_retrievers.json").write_text(out["extra_retrievers.json"])
    print("wrote", here / "results" / "extra_retrievers.json")
