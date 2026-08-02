# FinRank baselines

Reproduces every number in the FinRank paper from the released
`FinRank.jsonl` alone.

```bash
pip install -r requirements.txt

python validate_release.py       --data ../FinRank.jsonl
python run_baselines.py          --data ../FinRank.jsonl --out results/
python run_extra_retrievers.py   --data ../FinRank.jsonl --out results/
python make_splits.py            --data ../FinRank.jsonl --out splits.json
python make_crosstabs.py         --data ../FinRank.jsonl --out ../
python make_macros.py            --data ../FinRank.jsonl \
    --results results/results.json \
    --repair-log ../repair_log.json --out _results_macros.tex
```

- `validate_release.py` — release gate: schema, closed vocabularies, duplicates,
  degenerate negatives, page formats, and external-source markers (exits
  non-zero on any violation).
- `run_baselines.py` — TF-IDF, BM25, all-mpnet-base-v2, cross-encoder
  reranking, metadata-filtered BM25, stratified metrics, query-rewrite
  ablation, hard-vs-random pairwise contrast (seed 42).
- `run_extra_retrievers.py` — bge-large-en-v1.5, a finance-adapted BGE, and
  e5-mistral-7b-instruct (encoded per its reference implementation:
  EOS-terminated inputs, last-token pooling). GPU recommended for the 7B.
- `make_splits.py` / `make_crosstabs.py` / `make_macros.py` — regenerate the
  released split assignments, the paper's appendix cross-tabulations, and the
  paper's numbers file, so no reported count can drift from the data.
- `modal_run.py` / `modal_extra.py` — optional Modal (GPU) wrappers for the
  two evaluation scripts (`modal run modal_run.py`).

Conventions (fixed to match the published numbers): Recall@k is the
per-record fraction of gold passages retrieved, macro-averaged; pairwise
accuracy is per-record with ties counting for the gold; the corpus is the
first-occurrence deduplication of all non-empty passage and hard-negative
texts (5,270 entries); all encoders truncate at 512 tokens. On a single
A10G GPU the full pipeline completes in under fifteen minutes.
