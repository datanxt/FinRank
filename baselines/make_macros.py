#!/usr/bin/env python3
"""Regenerate the paper's _results_macros.tex from the released data and
baselines/results/results.json.

Every macro in the file is recomputed here — data-derived counts directly from
FinRank.jsonl (+ repair_log.json for the affected-record counts), model metrics
from results.json produced by run_baselines.py.

Usage:
  python make_macros.py --data FinRank.jsonl --results results/results.json \
      --repair-log repair_log.json --out _results_macros.tex
"""
import argparse, json, collections
import numpy as np

MODEL_GLOBAL = {"bm25": "BM", "tfidf": "Tfidf", "dense": "Dense", "ce_dense": "CeDense", "ce_inrec": "CeInrec"}
MODEL_ABL = {"bm25": "BmTwofive", "tfidf": "Tfidf", "dense": "Dense", "ce_dense": "CeDenseTopK", "ce_inrec": "CeInrec"}
BUCKET = {
    "doc_type": ("Doctype", {"10-K": "TenK", "10-Q": "TenQ", "Unknown": "Unknown"}),
    "reasoning_type": ("Reasoningtype", {"Qualitative": "Qualitative", "Quantitative": "Quantitative", "Unknown": "Unknown"}),
    "passage_type": ("Passagetype", {"Single-Passage": "Singlepassage", "Two-Passage": "Twopassage",
                                     "Multi-Passage": "Multipassage", "Unknown": "Unknown"}),
    "difficulty": ("Difficulty", {"Easy": "Easy", "Medium": "Medium", "Hard": "Hard", "Unknown": "Unknown"}),
}
QR_MODELS = {"bm25": "BmTwofive", "tfidf": "Tfidf", "dense": "Dense"}


def fmt(v, nd=1):
    if v is None:
        return "--"
    return f"{v:.2f}" if abs(v) < 0.1 and v != 0 else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--repair-log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    res = json.load(open(args.results))
    rlog = json.load(open(args.repair_log))

    M = {}  # macro name -> string value

    # ---------------- data-derived ----------------
    n = len(recs)
    M["CorpusRecords"] = str(n)
    M["CorpusEntries"] = str(res["corpus_size"])
    M["CorpusGoldMean"] = f"{np.mean([len(r['passages']) for r in recs]):.2f}"
    M["CorpusHnMean"] = f"{np.mean([len(r['hard_negatives']) for r in recs]):.2f}"

    def tally(field, mapping):
        c = collections.Counter(str(r.get(field, "Unknown")) for r in recs)
        return {mk: c.get(bk, 0) for bk, mk in mapping.items()}

    for f, name, mapping in [
        ("difficulty", "Difficulty", {"Easy": "Easy", "Medium": "Medium", "Hard": "Hard", "Unknown": "Unknown"}),
        ("industry", "Industry", {"Pharmaceuticals": "Pharmaceuticals", "Automotive": "Automotive", "Oil & Gas": "Oilgas"}),
        ("topic", "Topic", {"Company Overview": "Companyoverview", "Financials": "Financials",
                            "Shareholder Return": "Shareholderreturn", "Accounting": "Accounting", "Legal": "Legal",
                            "Governance": "Governance", "Risk": "Risk", "Footnotes": "Footnotes"}),
        ("reasoning_type", "Reasoningtype", {"Qualitative": "Qualitative", "Quantitative": "Quantitative", "Unknown": "Unknown"}),
        ("passage_type", "Passagetype", {"Single-Passage": "Singlepassage", "Multi-Passage": "Multipassage",
                                         "Two-Passage": "Twopassage", "Unknown": "Unknown"}),
        ("doc_type", "Doctype", {"10-K": "TenK", "10-Q": "TenQ", "Unknown": "Unknown"}),
        ("year", "Year", {"2025": "Twentyfive", "Unknown": "Unknown", "2024": "Twentyfour"}),
    ]:
        for mk, v in tally(f, mapping).items():
            M[f"Norm{name}{mk}"] = str(v)

    skip_rules = {"year-to-int"}
    affected = collections.defaultdict(set)
    for c in rlog["changes"]:
        if c["rule"] in skip_rules:
            continue
        f = c["field"].split("[")[0].split(".")[0]
        affected[f].add(c["id"])
    for f, name in [("difficulty", "Difficulty"), ("industry", "Industry"), ("topic", "Topic"),
                    ("reasoning_type", "Reasoningtype"), ("passage_type", "Passagetype"),
                    ("doc_type", "Doctype"), ("year", "Year")]:
        M[f"NormAffected{name}"] = str(len(affected.get(f, set())))

    # taxonomy (same classification as make_taxonomy.py)
    smap = {r["ticker"]: r["industry"] for r in recs}
    tax = collections.Counter()
    pos_sha = collections.defaultdict(list)
    for r in recs:
        for p in r["passages"]:
            pos_sha[p.get("text_sha1")].append(r["id"])
    contam_hn, contam_recs = 0, set()
    for r in recs:
        for h in r["hard_negatives"]:
            ht = str(h.get("ticker", "")).strip()
            if ht == r["ticker"]:
                same = str(h.get("year", "")).strip() == str(r["year"]) and str(h.get("doc_type", "")).strip() == str(r["doc_type"])
                tax["Samecompanysameyeardoc" if same else "Samecompanyotheryearordoc"] += 1
            elif smap.get(ht) == r["industry"]:
                tax["Sameindustryothercompany"] += 1
            else:
                tax["Crossindustry"] += 1
            if h.get("text_sha1") in pos_sha:
                contam_hn += 1
                contam_recs.add(r["id"])
    total = sum(tax.values())
    M["TaxTotal"] = str(total)
    for k in ("Sameindustryothercompany", "Samecompanyotheryearordoc", "Samecompanysameyeardoc", "Crossindustry"):
        M[f"Tax{k}"] = str(tax[k])
        M[f"TaxPct{k}"] = fmt(100 * tax[k] / total)
    M["TaxContamRecords"] = str(len(contam_recs))
    M["TaxContamHN"] = str(contam_hn)

    # ---------------- model metrics ----------------
    key_map = [("RecallOne", "recall@1"), ("RecallFive", "recall@5"), ("RecallTen", "recall@10"),
               ("RecallTwenty", "recall@20"), ("MRR", "mrr"), ("NDCG", "ndcg@10")]
    for m, pref in MODEL_GLOBAL.items():
        mm = res["models"].get(m, {})
        for suf, k in key_map:
            M[f"{pref}{suf}"] = fmt(mm.get(k))
        M[f"{pref}InRecMRR"] = fmt(mm.get("inrec_mrr"))
        M[f"{pref}InRecNDCGFive"] = fmt(mm.get("inrec_ndcg@5"))
        M[f"{pref}HnAcc"] = fmt(mm.get("hn_acc"))
        M[f"{pref}RandAcc"] = fmt(mm.get("rand_acc"))

    # stratified: global metrics from results['strata']; CeInrec stratified in-record MRR
    inrec_pr = res.get("inrec_per_record", {}).get("ce_inrec", [])
    for field, (fname, mapping) in BUCKET.items():
        strat = res["strata"].get(field, {})
        for bk, bname in mapping.items():
            for m, mpref in MODEL_ABL.items():
                base = f"Abl{fname}{bname}{mpref}"
                if m == "ce_inrec":
                    ix = [i for i, r in enumerate(recs) if str(r.get(field, "Unknown")) == bk
                          and i < len(inrec_pr) and inrec_pr[i] is not None]
                    M[f"{base}RecallTen"] = "--"
                    M[f"{base}NDCG"] = "--"
                    M[f"{base}MRR"] = fmt(100 * float(np.mean([inrec_pr[i] for i in ix]))) if ix else "--"
                else:
                    sm = strat.get(bk, {}).get(m)
                    M[f"{base}RecallTen"] = fmt(sm and sm.get("recall@10"))
                    M[f"{base}NDCG"] = fmt(sm and sm.get("ndcg@10"))
                    M[f"{base}MRR"] = fmt(sm and sm.get("mrr"))

    # query-rewrite ablation
    for m, mpref in QR_MODELS.items():
        qr = res["query_rewrite"].get(m, {})
        for cond, cname in [("raw", "Raw"), ("with_rewrites", "Withrewrites")]:
            for suf, k in [("RecallTen", "recall@10"), ("NDCG", "ndcg@10"), ("MRR", "mrr")]:
                M[f"QR{cname}{mpref}{suf}"] = fmt(qr.get(cond, {}).get(k))

    # metadata-filtered BM25
    orc = res.get("metadata_filtered", res.get("metadata_oracle", {}))
    for suf, k in [("RecallOne", "recall@1"), ("RecallFive", "recall@5"), ("RecallTen", "recall@10"),
                   ("MRR", "mrr"), ("NDCG", "ndcg@10")]:
        M[f"MetaFilterBm{suf}"] = fmt(orc.get("bm25", {}).get(k))
    M["MetaFilterHnElimPct"] = fmt(orc.get("hn_eliminated_pct"))
    M["MetaFilterMeanCands"] = fmt(orc.get("mean_candidates"), 0)
    M["MetaFilterGoldLossSome"] = str(orc.get("records_losing_any_gold", "--"))
    M["MetaFilterGoldLossAll"] = str(orc.get("records_losing_all_golds", "--"))

    # split sizes from the released splits.json (sibling of the results dir)
    import os as _os
    sp_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(args.results))), "splits.json")
    if _os.path.exists(sp_path):
        sp = json.load(open(sp_path))
        M["SplitRandomTrain"] = str(len(sp["random"]["train"]))
        M["SplitRandomDev"] = str(len(sp["random"]["dev"]))
        M["SplitRandomTest"] = str(len(sp["random"]["test"]))
        M["SplitTickerTrain"] = str(len(sp["ticker"]["train"]))
        M["SplitTickerDev"] = str(len(sp["ticker"]["dev"]))
        M["SplitTickerTest"] = str(len(sp["ticker"]["test"]))
        M["SplitYearTest"] = str(len(sp["year"]["test"]))
        M["SplitDoctypeTest"] = str(len(sp["doctype"]["test"]))

    # query-rewrite coverage
    qr_n = sum(1 for r in recs if r.get("query_rewrite"))
    M["QRCount"] = str(qr_n)
    M["QRCoveragePct"] = f"{100*qr_n/len(recs):.0f}"

    # additional dense retrievers (results/extra_retrievers.json), if present
    import os
    extra_path = os.path.join(os.path.dirname(args.results), "extra_retrievers.json")
    EXTRA = {"fin_bge": "FinBge", "bge_large": "BgeLarge", "e5_mistral": "EFiveMistral"}
    extra = json.load(open(extra_path)) if os.path.exists(extra_path) else {}
    for mk, pref in EXTRA.items():
        mm = extra.get(mk, {})
        for suf, k in key_map:
            M[f"{pref}{suf}"] = fmt(mm.get(k))
        M[f"{pref}InRecMRR"] = fmt(mm.get("inrec_mrr"))
        M[f"{pref}InRecNDCGFive"] = fmt(mm.get("inrec_ndcg@5"))
        M[f"{pref}HnAcc"] = fmt(mm.get("hn_acc"))
        M[f"{pref}RandAcc"] = fmt(mm.get("rand_acc"))

    with open(args.out, "w") as f:
        f.write("% Auto-generated by baselines/make_macros.py. Do not edit by hand.\n")
        f.write("% Re-run run_baselines.py then make_macros.py after any data change.\n\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {args.out}: {len(M)} macros")


if __name__ == "__main__":
    main()
