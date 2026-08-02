# FinRank

**An Evidence-Grounded Benchmark for Financial Question Answering and Retrieval over SEC Filings**

FinRank is a benchmark of **1,185 manually authored question–answer records** over the 10-K and 10-Q filings (2024–2025) of **22 US companies** in three sectors (pharmaceuticals, oil & gas, automotive). Each record pairs a question and reference answer with:

- **gold supporting passages** transcribed from the source filing (mean 1.96 per record),
- **hand-curated hard negatives** (6,021 total; mean 5.08 per record) drawn from confusable passages — within the same filing, across reporting periods of the same firm, and across comparable firms,
- rich metadata: topic (8 categories), difficulty (Easy/Medium/Hard), reasoning type (qualitative/quantitative), evidence scope (single/two/multi-passage), document type, filing year,
- gold sub-question decompositions (`query_rewrite`) for 69% of records.

FinRank is, to our knowledge, the first financial QA benchmark to release per-question curated hard negatives and define explicit **reranking** and **hard-negative discrimination** tasks alongside passage retrieval.

## Hard-negative taxonomy

| Relationship to source record | n | share |
|---|---|---|
| Same industry, different company | 4,807 | 79.8% |
| Same company, different year/form | 794 | 13.2% |
| Same company, same year & form | 419 | 7.0% |
| Cross-industry | 1 | <0.1% |

442 hard negatives (7.3%) are whitespace-identical to a supporting passage of a *different* record (recurring boilerplate that may legitimately answer more than one question); `hn_taxonomy.json` carries per-passage overlap flags so users can filter them.

## Files

| File | Description |
|---|---|
| `FinRank.jsonl` | The dataset: 1,185 records, one JSON object per line, canonical normalized labels. Every passage and hard negative carries a stable `passage_id` and `text_sha1`; wherever a label was normalized during curation, the original surface form is preserved inline in a parallel `<field>_raw` field. |
| `repair_log.json` | Machine-readable curation log: every normalization and repair applied in producing the release (each entry carries record id, field, old/new value, and rule), including all record-level exclusions, plus 56 records whose stored labels disagree with their structured `question_id`, flagged for audit. |
| `hn_taxonomy.json` | Per-hard-negative taxonomy category and cross-record overlap flags. |
| `summary.json` | Label tallies and cross-tabulations of the release. |
| `baselines/` | Evaluation harness (`run_baselines.py`), release validator (`validate_release.py`), split assignments (`splits.json`), macro generator, and the reported results. |

## Evaluation regimes

- **Passage retrieval** over the global pooled corpus *C* — the union of all supporting-passage and hard-negative texts (5,230 unique passages). Note that *C* is a *curated passage-ranking stress test* (annotated positives + curated distractors), **not** the full text of the filings; scores are not comparable to full-document retrieval.
- **Reranking / hard-negative discrimination** over the in-record candidate set (gold passages ∪ hard negatives), reporting MRR, nDCG@5, and pairwise accuracy.

## Provenance

Questions, answers, gold passages, and hard negatives were authored fully manually by five business students working from the filings under a shared question-generation criteria document, with sampled instructor review during collection. Deterministic post-hoc curation (label normalization, recovery of placeholder fields from the structured question IDs, hard-negative metadata repair, removal of degenerate and within-record duplicate passages, and exclusion of 65 records (52 relying on a non-filing academic source, 3 with synthetic relevance-rationale negatives, 3 with placeholder content, 4 duplicate question–answer pairs, and 3 with insufficient hard negatives)) is fully documented in `repair_log.json`. See the paper for construction criteria, known limitations (single annotator per record, no formal inter-annotator agreement), and the target-versus-realized distribution audit.

## License

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see `LICENSE`). Academic and educational use permitted with attribution; commercial licensing available from the authors on request. The underlying SEC filings are public records; FinRank redistributes only derived question–answer records, supporting passages, and hard negatives. The dataset contains no non-public personal information.

## Citation

Paper under review; citation entry will be added upon posting. Until then, please cite this repository:

```bibtex
@misc{finrank2026,
  title  = {FinRank: An Evidence-Grounded Benchmark for Financial Question Answering and Retrieval over SEC Filings},
  author = {Mansouri, Sasan and Saad, Daniel and Wahrenburg, Mark and Weissel, Manu and Woebbeking, Fabian},
  year   = {2026},
  url    = {https://github.com/datanxt/FinRank}
}
```
