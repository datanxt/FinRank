#!/usr/bin/env python3
"""Generate the five released generalization splits as record-ID lists.

Splits (all deterministic, seed 42):
  random  : 80/10/10 train/dev/test over shuffled record ids
  ticker  : test = records of held-out tickers (smallest tickers accumulated
            until >= 10% of records), dev likewise from the next tickers
  sector  : three leave-one-sector-out folds (test = the held-out sector)
  year    : test = filing year 2024, train = 2025 (Unknown-doc records in train)
  doctype : test = 10-Q, train = 10-K (Unknown in train)

Usage: python make_splits.py --data FinRank.jsonl --out splits.json
"""
import argparse, json, random, collections

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    ids = [r["id"] for r in recs]
    n = len(ids)
    splits = {}

    rng = random.Random(42)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    a, b = int(0.8 * n), int(0.9 * n)
    splits["random"] = {"train": sorted(shuffled[:a]), "dev": sorted(shuffled[a:b]),
                        "test": sorted(shuffled[b:])}

    by_ticker = collections.defaultdict(list)
    for r in recs:
        by_ticker[r["ticker"]].append(r["id"])
    order = sorted(by_ticker, key=lambda t: (len(by_ticker[t]), t))
    test_t, dev_t, cnt = [], [], 0
    it = iter(order)
    for t in it:
        test_t.append(t); cnt += len(by_ticker[t])
        if cnt >= 0.10 * n:
            break
    cnt = 0
    for t in it:
        dev_t.append(t); cnt += len(by_ticker[t])
        if cnt >= 0.10 * n:
            break
    train_t = [t for t in by_ticker if t not in test_t + dev_t]
    splits["ticker"] = {
        "held_out_test_tickers": test_t, "held_out_dev_tickers": dev_t,
        "train": sorted(i for t in train_t for i in by_ticker[t]),
        "dev": sorted(i for t in dev_t for i in by_ticker[t]),
        "test": sorted(i for t in test_t for i in by_ticker[t]),
    }

    by_sector = collections.defaultdict(list)
    for r in recs:
        by_sector[r["industry"]].append(r["id"])
    splits["sector"] = {f"test_{s.replace(' ', '_')}": {
        "train": sorted(i for s2, v in by_sector.items() if s2 != s for i in v),
        "test": sorted(v)} for s, v in by_sector.items()}

    splits["year"] = {"train": sorted(r["id"] for r in recs if r.get("year") != 2024),
                      "test": sorted(r["id"] for r in recs if r.get("year") == 2024)}
    splits["doctype"] = {"train": sorted(r["id"] for r in recs if r.get("doc_type") != "10-Q"),
                         "test": sorted(r["id"] for r in recs if r.get("doc_type") == "10-Q")}

    json.dump(splits, open(args.out, "w"), indent=1)
    sizes = {k: {kk: (len(vv) if isinstance(vv, list) else {k3: len(v3) for k3, v3 in vv.items()})
                 for kk, vv in v.items() if not kk.startswith("held_out")}
             for k, v in splits.items()}
    print(json.dumps(sizes, indent=1))


if __name__ == "__main__":
    main()
