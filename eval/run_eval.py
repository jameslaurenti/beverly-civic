#!/usr/bin/env python3
"""
Runs the eval set against the Beverly Civic Assistant and scores results.

Reads:  eval/eval_set.json
Writes: eval/eval_results.json

Pass/fail is determined by whether the expected source title appears in the
returned source titles (case-insensitive substring match).

Run:
    python eval/run_eval.py                                       # prod
    python eval/run_eval.py --app-url http://localhost:8000       # local
    python eval/run_eval.py --limit 10                            # first 10 only
    python eval/run_eval.py --data-types calendar news            # filter by type
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"
DEFAULT_APP_URL = "https://beverly-civic.onrender.com"


def ask(app_url: str, question: str) -> tuple[str, list[dict]]:
    resp = requests.post(
        f"{app_url}/ask",
        json={"text": question, "history": []},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("answer", ""), data.get("sources", [])


def _source_hit(expected_title: str, source_titles: list[str]) -> bool:
    if not expected_title:
        return True
    expected_lower = expected_title.lower()
    return any(expected_lower in t.lower() for t in source_titles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument("--limit", type=int, help="Run only first N questions")
    parser.add_argument("--data-types", nargs="+",
                        help="Filter by data type (calendar news budget minutes library)")
    args = parser.parse_args()

    if not EVAL_SET_PATH.exists():
        raise SystemExit(f"Eval set not found at {EVAL_SET_PATH}. Run generate_eval.py first.")

    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))

    if args.data_types:
        eval_set = [q for q in eval_set if q.get("data_type") in args.data_types]
    if args.limit:
        eval_set = eval_set[:args.limit]

    log.info("Running %d questions against %s", len(eval_set), args.app_url)

    results = []
    by_type: dict[str, list[bool]] = {}

    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        dtype = item.get("data_type", "?")
        log.info("[%d/%d] %s", i, len(eval_set), q)
        try:
            actual_answer, sources = ask(args.app_url, q)
            source_titles = [s.get("title", "") for s in sources]
            passed = _source_hit(item.get("source_title", ""), source_titles)
            result = {
                **item,
                "actual_answer": actual_answer,
                "sources_returned": source_titles,
                "pass": passed,
            }
            results.append(result)
            status = "PASS" if passed else "FAIL"
            log.info("  %s | top: %s", status, source_titles[0] if source_titles else "none")
            by_type.setdefault(dtype, []).append(passed)
        except Exception as e:
            log.warning("  ERROR: %s", e)
            results.append({**item, "actual_answer": f"ERROR: {e}",
                            "sources_returned": [], "pass": False})
            by_type.setdefault(dtype, []).append(False)
        if i < len(eval_set):
            time.sleep(1)  # be polite to Render free tier

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    total = len(results)
    overall = sum(r.get("pass", False) for r in results)
    print(f"OVERALL: {overall}/{total} ({100 * overall // total}%)")
    print("-" * 60)
    for dtype, hits in sorted(by_type.items()):
        n = len(hits)
        p = sum(hits)
        print(f"  {dtype:10s}  {p}/{n} ({100 * p // n}%)")
    print("-" * 60)
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        print(f"  {status} [{r.get('data_type','?'):8s}] {r['question'][:70]}")
    print(f"\nFull results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
