#!/usr/bin/env python3
"""Additional dense retrievers on FinRank, using the exact metric conventions
of run_baselines.py (same corpus construction, same seed-42 random negatives,
same per-record pairwise-accuracy definition).

Models: a finance-adapted embedder, a strong general embedder, and a 7B
instruction-tuned embedder (the strongest dense model evaluated by FinDER).

Usage: python run_extra_retrievers.py --data FinRank.jsonl --out results/
"""
import argparse, json
from pathlib import Path
import numpy as np

from run_baselines import (load_records, build_corpus, per_record_metrics,
                           inrec_metrics, pairwise_acc, aggregate, SEED)

MODELS = {
    "fin_bge": {
        "id": "FinLang/finance-embeddings-investopedia",
        "q_prefix": "", "batch": 64,
    },
    "bge_large": {
        "id": "BAAI/bge-large-en-v1.5",
        "q_prefix": "Represent this sentence for searching relevant passages: ",
        "batch": 64,
    },
    "e5_mistral": {
        "id": "intfloat/e5-mistral-7b-instruct",
        "q_prefix": ("Instruct: Given a financial question, retrieve passages "
                     "from SEC filings that answer it\nQuery: "),
        "batch": 4,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="fin_bge,bge_large,e5_mistral")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    recs = load_records(args.data)
    texts, meta, index = build_corpus(recs)
    golds = [{index[p["text"]] for p in r["passages"] if str(p["text"]).strip()} for r in recs]
    hns = [[index[h["text"]] for h in r["hard_negatives"] if str(h["text"]).strip()] for r in recs]
    queries = [r["question"] for r in recs]
    print(f"records={len(recs)} corpus={len(texts)}")

    rng = np.random.default_rng(SEED)
    rand_negs = []
    for i in range(len(recs)):
        pool = np.setdiff1d(np.arange(len(texts)), np.array(sorted(golds[i]), dtype=int))
        rand_negs.append(rng.choice(pool, size=max(len(hns[i]), 1), replace=False))

    from sentence_transformers import SentenceTransformer
    import torch

    def encode_e5_reference(all_texts, batch):
        """e5-mistral per its model card: EOS-terminated inputs, last-token pool."""
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MODELS["e5_mistral"]["id"])
        mdl = AutoModel.from_pretrained(MODELS["e5_mistral"]["id"],
                                        torch_dtype=torch.bfloat16).to(device).eval()
        out = []
        for i in range(0, len(all_texts), batch):
            bd = tok(all_texts[i:i + batch], max_length=511, truncation=True,
                     padding=False, return_attention_mask=False)
            bd["input_ids"] = [ids + [tok.eos_token_id] for ids in bd["input_ids"]]
            bd = tok.pad(bd, padding=True, return_attention_mask=True,
                         return_tensors="pt").to(device)
            with torch.no_grad():
                h = mdl(**bd).last_hidden_state
            sl = bd["attention_mask"].sum(dim=1) - 1
            emb = h[torch.arange(h.shape[0], device=device), sl]
            out.append(F.normalize(emb, p=2, dim=1).float().cpu().numpy())
        del mdl
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return np.concatenate(out)

    out = {}
    for key in args.models.split(","):
        cfg = MODELS[key]
        print(f"=== {key}: {cfg['id']}")
        if key == "e5_mistral":
            E_c = encode_e5_reference(texts, cfg["batch"])
            E_q = encode_e5_reference([cfg["q_prefix"] + q for q in queries], cfg["batch"])
        else:
            enc = SentenceTransformer(cfg["id"], trust_remote_code=True)
            enc.max_seq_length = 512  # uniform truncation across all evaluated encoders
            E_c = enc.encode(texts, batch_size=cfg["batch"], normalize_embeddings=True,
                             show_progress_bar=True)
            E_q = enc.encode([cfg["q_prefix"] + q for q in queries], batch_size=cfg["batch"],
                             normalize_embeddings=True)
            del enc
        S = (E_q @ E_c.T).astype(np.float32)

        rk = np.argsort(-S, axis=1)
        rows = [per_record_metrics(rk[i], golds[i]) for i in range(len(recs))]
        res = {k: aggregate(rows, k) for k in rows[0]}

        mrrs, ndcgs, hn_accs, rd_accs = [], [], [], []
        for i in range(len(recs)):
            g = sorted(golds[i])
            if not g:
                continue
            sg = S[i, g]
            if hns[i]:
                sh = S[i, hns[i]]
                mrr, nd = inrec_metrics(sg, sh)
                mrrs.append(mrr); ndcgs.append(nd)
                hn_accs.append(pairwise_acc(sg, sh)[0])
            rd_accs.append(pairwise_acc(sg, S[i, list(rand_negs[i])])[0])
        res.update({
            "inrec_mrr": 100 * float(np.mean(mrrs)),
            "inrec_ndcg@5": 100 * float(np.mean(ndcgs)),
            "hn_acc": 100 * float(np.mean(hn_accs)),
            "rand_acc": 100 * float(np.mean(rd_accs)),
        })

        strata = {}
        for field in ("doc_type", "reasoning_type", "passage_type", "difficulty"):
            buckets = {}
            for i, r in enumerate(recs):
                buckets.setdefault(str(r.get(field, "Unknown")), []).append(i)
            strata[field] = {b: aggregate([rows[i] for i in ix], "recall@10")
                             for b, ix in buckets.items()}
        res["strata_recall@10"] = strata
        out[key] = res
        print(key, {k: round(v, 1) for k, v in res.items() if isinstance(v, float)})
        json.dump(out, (outdir / "extra_retrievers.json").open("w"), indent=1)  # save per model
    print("wrote", outdir / "extra_retrievers.json")


if __name__ == "__main__":
    main()
