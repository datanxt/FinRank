#!/usr/bin/env python3
"""Release-blocking validation for FinRank.jsonl. Exits non-zero on any violation.

Checks: schema completeness and closed label vocabularies; no placeholder
values; no empty passage or hard-negative texts; passage_type and num_passages
consistent with the gold-passage count; no duplicate question-answer pairs; no
hard negative identical to a gold passage of its own record; a minimum
hard-negative count per record; no external (non-SEC) source markers; unique
record and passage IDs.

Usage: python validate_release.py --data FinRank.jsonl
"""
import argparse, json, sys, collections

TOPICS = {"Company Overview", "Financials", "Footnotes", "Governance",
          "Accounting", "Legal", "Risk", "Shareholder Return"}
SECTORS = {"Pharmaceuticals", "Oil & Gas", "Automotive"}
DIFF = {"Easy", "Medium", "Hard"}
REAS = {"Qualitative", "Quantitative"}
PTYPE = {"Single-Passage", "Two-Passage", "Multi-Passage"}
DOC = {"10-K", "10-Q", "Unknown"}
EXTERNAL = ["10.1111/opec", "wiley online library", "shibboleth",
            "onlinelibrary.wiley", "opec energy review", "academic review",
            "model b.3", "cogsta", "capexsa", "ltdtc", "holding period return",
            "sciencedirect", "springer.com", "doi.org/10."]
MIN_HN = 4
FN_LEAK = __import__("re").compile(r'[A-Z]{1,6}-10-[KQ]-20\d{2}(?:-Q[1-4])?\.md')
RAT_MARK = __import__("re").compile(r"\[\d+\]")
RAT_PHRASES = ["pertains to a different", "does not specifically address", "does not address",
               "not directly address", "rather than", "unrelated to the", "but is about",
               "is about a different", "not the same"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    errs = []

    ids = collections.Counter(r["id"] for r in recs)
    for k, v in ids.items():
        if v > 1:
            errs.append(f"duplicate record id {k}")
    qa = collections.Counter((r["question"].strip(), r["answer"].strip()) for r in recs)
    for (q, _), v in qa.items():
        if v > 1:
            errs.append(f"duplicate question-answer pair: {q[:60]!r}")

    pids = set()
    for r in recs:
        rid = r["id"][:8]
        for f, vocab in [("topic", TOPICS), ("industry", SECTORS), ("difficulty", DIFF),
                         ("reasoning_type", REAS), ("passage_type", PTYPE), ("doc_type", DOC)]:
            if r.get(f) not in vocab:
                errs.append(f"{rid}: {f}={r.get(f)!r} outside vocabulary")
        if not isinstance(r.get("year"), int) or not 2000 <= r["year"] <= 2030:
            errs.append(f"{rid}: invalid year {r.get('year')!r}")
        if str(r.get("ticker", "")).strip().lower() in ("", "string"):
            errs.append(f"{rid}: placeholder ticker")
        for x in r.get("query_rewrite") or []:
            if not str(x).strip() or str(x).strip().lower() == "string":
                errs.append(f"{rid}: placeholder in query_rewrite")

        n = len(r.get("passages") or [])
        if n == 0:
            errs.append(f"{rid}: no gold passages")
        want = "Single-Passage" if n == 1 else "Two-Passage" if n == 2 else "Multi-Passage"
        if r.get("passage_type") != want:
            errs.append(f"{rid}: passage_type={r.get('passage_type')} but {n} golds")
        if r.get("num_passages") != n:
            errs.append(f"{rid}: num_passages={r.get('num_passages')} but {n} golds")
        if len(r.get("hard_negatives") or []) < MIN_HN:
            errs.append(f"{rid}: only {len(r.get('hard_negatives') or [])} hard negatives (< {MIN_HN})")

        for h in r.get("hard_negatives") or []:
            if str(h.get("doc_type", "")).strip() not in ("10-K", "10-Q"):
                errs.append(f"{rid}: hard-negative doc_type {h.get('doc_type')!r} outside 10-K/10-Q")
        gold = {str(p["text"]).strip() for p in r["passages"]}
        for i, h in enumerate(r.get("hard_negatives") or []):
            if not str(h.get("text", "")).strip():
                errs.append(f"{rid}: empty hard-negative text [{i}]")
            elif str(h["text"]).strip() in gold:
                errs.append(f"{rid}: hard negative [{i}] duplicates a gold passage")
        for p in (r.get("passages") or []) + (r.get("hard_negatives") or []):
            if not str(p.get("text", "")).strip():
                errs.append(f"{rid}: empty passage text")
            pid = p.get("passage_id")
            if not pid or pid in pids:
                errs.append(f"{rid}: missing/duplicate passage_id {pid}")
            pids.add(pid)
            blob = str(p.get("text", "")).lower()
            for m in EXTERNAL:
                if m in blob:
                    errs.append(f"{rid}: external source marker {m!r} in passage text")
        if str(r.get("question", "")).strip().lower() in ("", "string") \
           or str(r.get("answer", "")).strip().lower() in ("", "string"):
            errs.append(f"{rid}: placeholder or empty question/answer text")
        for p3 in (r.get("passages") or []) + (r.get("hard_negatives") or []):
            t3 = str(p3.get("text", ""))
            if t3.strip().lower() == "string":
                errs.append(f"{rid}: placeholder passage text")
            if FN_LEAK.search(t3):
                errs.append(f"{rid}: annotation filename leaked in passage text")
            if len(t3) < 700 and RAT_MARK.search(t3) and any(x in t3.lower() for x in RAT_PHRASES):
                errs.append(f"{rid}: synthetic relevance-rationale text in passage/negative")
        blob = (r["question"] + " " + r["answer"] + " "
                + " ".join(str(x) for x in r.get("query_rewrite") or [])).lower()
        for m in EXTERNAL:
            if m in blob:
                errs.append(f"{rid}: external source marker {m!r} in question/answer/rewrite")
        gtexts = [str(p["text"]).strip() for p in r.get("passages") or []]
        if len(gtexts) != len(set(gtexts)):
            errs.append(f"{rid}: duplicate gold passage text within record")
        htexts = [str(h["text"]).strip() for h in r.get("hard_negatives") or []]
        if len(htexts) != len(set(htexts)):
            errs.append(f"{rid}: duplicate hard-negative text within record")
        if len(set(htexts)) < MIN_HN:
            errs.append(f"{rid}: fewer than {MIN_HN} unique hard negatives")
        import re as _re
        for p2 in (r.get("passages") or []) + (r.get("hard_negatives") or []):
            pv = p2.get("page_number")
            if pv is not None and str(pv).strip() and not _re.fullmatch(r"\d+([\u2013-]\d+)?", str(pv).strip()):
                errs.append(f"{rid}: invalid page_number {pv!r}")

    print(f"validated {len(recs)} records: {len(errs)} violation(s)")
    for e in errs[:40]:
        print(" -", e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
