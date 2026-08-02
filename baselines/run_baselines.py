#!/usr/bin/env python3
"""FinRank retrieval / reranking / hard-negative-discrimination baselines.

Reproduces every model-derived number in the paper from the released
FinRank.jsonl alone:

  * global retrieval over the pooled corpus C (TF-IDF, BM25, dense bi-encoder,
    cross-encoder reranking of dense top-20)
  * in-record reranking over L_r = gold ∪ hard negatives (all models + CE)
  * pairwise accuracy on (gold, hard-negative) and (gold, random-negative)
    pairs (seed 42)
  * stratified metrics (document type, reasoning type, passage type, difficulty)
  * query-rewrite ablation (question alone vs. question + rewrites)
  * metadata-filter oracle: BM25 restricted to corpus entries whose
    (ticker, year, doc_type) match the query record

Usage:
  python run_baselines.py --data FinRank.jsonl --out results/ [--skip-neural]

Outputs results/results.json and results/ablations.json. Feed results.json to
make_macros.py to regenerate the paper's _results_macros.tex.
"""
import argparse, json, re, collections
from pathlib import Path
import numpy as np

DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOPK_RERANK = 20
SEED = 42


def tokenize(t):
    return re.findall(r"\w+", str(t).lower())


def load_records(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_corpus(recs):
    """Unique non-empty texts, first-occurrence order and metadata."""
    texts, meta, index = [], [], {}
    for r in recs:
        for p in r["passages"]:
            t = p["text"]
            if not str(t).strip() or t in index:
                continue
            index[t] = len(texts)
            texts.append(t)
            meta.append((str(r.get("ticker", "")), str(r.get("year", "")), str(r.get("doc_type", ""))))
        for h in r["hard_negatives"]:
            t = h["text"]
            if not str(t).strip() or t in index:
                continue
            index[t] = len(texts)
            texts.append(t)
            meta.append((str(h.get("ticker", "")), str(h.get("year", "")), str(h.get("doc_type", ""))))
    return texts, meta, index


def per_record_metrics(ranking, gold):
    """ranking: array of corpus ids best-first; gold: set of ids."""
    out = {}
    hits = np.isin(ranking, list(gold))
    for k in (1, 5, 10, 20):
        out[f"recall@{k}"] = hits[:k].sum() / len(gold)
    first = np.flatnonzero(hits)
    out["mrr"] = 1.0 / (first[0] + 1) if len(first) else 0.0
    dcg = sum(1.0 / np.log2(r + 2) for r in np.flatnonzero(hits[:10]))
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(gold), 10)))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def inrec_metrics(scores_gold, scores_neg):
    """Rank golds among gold ∪ negatives by score; MRR + nDCG@5."""
    all_scores = np.concatenate([scores_gold, scores_neg])
    is_gold = np.zeros(len(all_scores), bool)
    is_gold[: len(scores_gold)] = True
    order = np.argsort(-all_scores, kind="stable")
    hits = is_gold[order]
    first = np.flatnonzero(hits)
    mrr = 1.0 / (first[0] + 1) if len(first) else 0.0
    dcg = sum(1.0 / np.log2(r + 2) for r in np.flatnonzero(hits[:5]))
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(int(is_gold.sum()), 5)))
    return mrr, (dcg / idcg if idcg else 0.0)


def pairwise_acc(scores_gold, scores_neg):
    """Per-record pairwise accuracy: fraction of (gold, neg) pairs with
    score(gold) >= score(neg). Ties count for the gold, matching the
    original FinRank evaluation convention (verified against the published
    numbers); records are averaged with equal weight by the caller."""
    wins = total = 0
    for g in scores_gold:
        for n in scores_neg:
            total += 1
            if g >= n:
                wins += 1
    return (wins / total, 1)


def aggregate(rows, key):
    return 100 * float(np.mean([r[key] for r in rows])) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-neural", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    recs = load_records(args.data)
    texts, meta, index = build_corpus(recs)
    print(f"records={len(recs)} corpus={len(texts)}")

    golds, hns = [], []
    for r in recs:
        golds.append({index[p["text"]] for p in r["passages"] if str(p["text"]).strip()})
        hns.append([index[h["text"]] for h in r["hard_negatives"] if str(h["text"]).strip()])
    queries = [r["question"] for r in recs]
    qr_mask = [bool(r.get("query_rewrite")) and any(str(x).strip() for x in r["query_rewrite"]) for r in recs]
    queries_qr = [
        q + " " + " ".join(str(x) for x in r.get("query_rewrite") or []) if m else q
        for q, r, m in zip(queries, recs, qr_mask)
    ]

    # ---------------- scoring backends ----------------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from rank_bm25 import BM25Okapi

    print("fitting tf-idf ...")
    vec = TfidfVectorizer(sublinear_tf=True)
    C_tfidf = vec.fit_transform(texts)

    def tfidf_scores(qs):
        return (vec.transform(qs) @ C_tfidf.T).toarray()

    print("fitting bm25 ...")
    bm25 = BM25Okapi([tokenize(t) for t in texts])

    def bm25_scores(qs):
        return np.stack([bm25.get_scores(tokenize(q)) for q in qs])

    cache = outdir / "score_cache.npz"
    if cache.exists():
        z = np.load(cache)
        score_mats = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
        qr_mats = {k[2:]: z[k] for k in z.files if k.startswith("q_")}
        print("loaded cached score matrices:", sorted(score_mats))
    else:
        score_mats = {"tfidf": tfidf_scores(queries), "bm25": bm25_scores(queries)}
        qr_mats = {"tfidf": tfidf_scores(queries_qr), "bm25": bm25_scores(queries_qr)}

    ce = None
    if not args.skip_neural:
        from sentence_transformers import SentenceTransformer, CrossEncoder

        if "dense" not in score_mats:
            print("encoding dense ...")
            enc = SentenceTransformer(DENSE_MODEL)
            E_c = enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
            E_q = enc.encode(queries, batch_size=64, normalize_embeddings=True)
            E_qr = enc.encode(queries_qr, batch_size=64, normalize_embeddings=True)
            score_mats["dense"] = (E_q @ E_c.T).astype(np.float32)
            qr_mats["dense"] = (E_qr @ E_c.T).astype(np.float32)
        ce = CrossEncoder(CE_MODEL)

    np.savez_compressed(cache, **{f"s_{k}": v.astype(np.float32) for k, v in score_mats.items()},
                        **{f"q_{k}": v.astype(np.float32) for k, v in qr_mats.items()})
    print("score matrices cached")

    rankings = {m: np.argsort(-s, axis=1) for m, s in score_mats.items()}

    # CE over dense top-20: rerank the head, keep dense order for the tail
    if ce is not None:
        print("cross-encoder reranking dense top-20 ...")
        ce_rankings = []
        for i, q in enumerate(queries):
            head = rankings["dense"][i][:TOPK_RERANK]
            s = ce.predict([(q, texts[j]) for j in head], show_progress_bar=False)
            ce_rankings.append(np.concatenate([head[np.argsort(-s, kind="stable")], rankings["dense"][i][TOPK_RERANK:]]))
        rankings["ce_dense"] = np.stack(ce_rankings)

    # ---------------- global + stratified ----------------
    results = {"n_records": len(recs), "corpus_size": len(texts), "models": {}}
    per_record = {}
    for m, rk in rankings.items():
        rows = [per_record_metrics(rk[i], golds[i]) for i in range(len(recs))]
        per_record[m] = rows
        results["models"][m] = {k: aggregate(rows, k) for k in rows[0]}
        print(m, {k: round(v, 1) for k, v in results["models"][m].items()})

    strata = {"doc_type": {}, "reasoning_type": {}, "passage_type": {}, "difficulty": {}}
    sizes = {}
    for field in strata:
        buckets = collections.defaultdict(list)
        for i, r in enumerate(recs):
            buckets[str(r.get(field, "Unknown"))].append(i)
        sizes[field] = {b: len(ix) for b, ix in buckets.items()}
        for b, ix in buckets.items():
            strata[field][b] = {
                m: {k: aggregate([per_record[m][i] for i in ix], k) for k in ("recall@10", "ndcg@10", "mrr")}
                for m in per_record
            }
    results["strata"] = strata

    # ---------------- in-record + pairwise ----------------
    rng = np.random.default_rng(SEED)
    rand_negs = []
    for i in range(len(recs)):
        pool = np.setdiff1d(np.arange(len(texts)), np.array(sorted(golds[i]), dtype=int))
        rand_negs.append(rng.choice(pool, size=max(len(hns[i]), 1), replace=False))

    results["inrec_per_record"] = {}

    def inrec_eval(name, score_fn):
        mrrs, ndcgs, hn_accs, rd_accs = [], [], [], []
        per_rec_mrr = [None] * len(recs)
        for i, r in enumerate(recs):
            g = sorted(golds[i])
            if not g:
                continue
            sg = score_fn(i, g)
            if hns[i]:
                sh = score_fn(i, hns[i])
                mrr, nd = inrec_metrics(sg, sh)
                mrrs.append(mrr)
                ndcgs.append(nd)
                per_rec_mrr[i] = mrr
                hn_accs.append(pairwise_acc(sg, sh)[0])
            rd_accs.append(pairwise_acc(sg, score_fn(i, list(rand_negs[i])))[0])
        results["inrec_per_record"][name] = per_rec_mrr
        results["models"].setdefault(name, {})
        results["models"][name].update({
            "inrec_mrr": 100 * float(np.mean(mrrs)),
            "inrec_ndcg@5": 100 * float(np.mean(ndcgs)),
            "hn_acc": 100 * float(np.mean(hn_accs)),
            "rand_acc": 100 * float(np.mean(rd_accs)),
        })
        print(name, "inrec", {k: round(results["models"][name][k], 1) for k in ("inrec_mrr", "inrec_ndcg@5", "hn_acc", "rand_acc")})

    for m in score_mats:
        inrec_eval(m, lambda i, ids, m=m: score_mats[m][i, ids])
    if ce is not None:
        print("cross-encoder in-record ...")
        ce_cache = {}

        def ce_score(i, ids):
            missing = [j for j in ids if (i, j) not in ce_cache]
            if missing:
                s = ce.predict([(queries[i], texts[j]) for j in missing], show_progress_bar=False)
                for j, v in zip(missing, s):
                    ce_cache[(i, j)] = float(v)
            return np.array([ce_cache[(i, j)] for j in ids])

        inrec_eval("ce_inrec", ce_score)

    # ---------------- query-rewrite ablation ----------------
    qr_ix = [i for i in range(len(recs)) if qr_mask[i]]
    results["query_rewrite"] = {"n": len(qr_ix)}
    for m, s in qr_mats.items():
        raw_rows = [per_record[m][i] for i in qr_ix]
        rk = np.argsort(-s[qr_ix], axis=1)
        qr_rows = [per_record_metrics(rk[j], golds[i]) for j, i in enumerate(qr_ix)]
        results["query_rewrite"][m] = {
            "raw": {k: aggregate(raw_rows, k) for k in ("recall@10", "ndcg@10", "mrr")},
            "with_rewrites": {k: aggregate(qr_rows, k) for k in ("recall@10", "ndcg@10", "mrr")},
        }

    # ---------------- metadata-filtered BM25 ----------------
    # Corpus entries carry first-occurrence metadata, so the filter can exclude
    # gold passages whose text first appeared under another filing's metadata;
    # the counts below quantify that provenance limitation.
    meta_arr = np.array(meta)
    elim, orows, cand_sizes = [], [], []
    lose_any = lose_all = 0
    bm = score_mats["bm25"]
    for i, r in enumerate(recs):
        key = (str(r.get("ticker", "")), str(r.get("year", "")), str(r.get("doc_type", "")))
        allowed = np.flatnonzero((meta_arr[:, 0] == key[0]) & (meta_arr[:, 1] == key[1]) & (meta_arr[:, 2] == key[2]))
        cand_sizes.append(len(allowed))
        if hns[i]:
            elim.append(1 - np.isin(np.array(hns[i]), allowed).mean())
        gids = np.array(sorted(golds[i]), dtype=int)
        kept_golds = np.isin(gids, allowed).sum()
        if kept_golds < len(gids):
            lose_any += 1
        if kept_golds == 0:
            lose_all += 1
        s = np.full(len(texts), -np.inf)
        s[allowed] = bm[i, allowed]
        orows.append(per_record_metrics(np.argsort(-s), golds[i]))
    results["metadata_filtered"] = {
        "bm25": {k: aggregate(orows, k) for k in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")},
        "hn_eliminated_pct": 100 * float(np.mean(elim)),
        "mean_candidates": float(np.mean(cand_sizes)),
        "records_losing_any_gold": lose_any,
        "records_losing_all_golds": lose_all,
    }
    print("metadata-filtered bm25:", {k: round(v, 1) for k, v in results["metadata_filtered"]["bm25"].items()},
          "| HN eliminated:", round(results["metadata_filtered"]["hn_eliminated_pct"], 1), "%",
          "| gold loss any/all:", lose_any, "/", lose_all)

    json.dump(results, (outdir / "results.json").open("w"), indent=1)
    json.dump(sizes, (outdir / "ablations.json").open("w"), indent=1)
    print("wrote", outdir / "results.json")


if __name__ == "__main__":
    main()
