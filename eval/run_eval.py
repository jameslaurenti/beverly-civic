#!/usr/bin/env python3
"""
Runs the eval set against the live Beverly Civic Assistant and scores results.

Reads:  eval/eval_set.json
Writes: eval/eval_results.json

Run:
    python eval/run_eval.py
    python eval/run_eval.py --app-url http://localhost:8000   # local app
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    args = parser.parse_args()

    if not EVAL_SET_PATH.exists():
        raise SystemExit(f"Eval set not found at {EVAL_SET_PATH}. Run generate_eval.py first.")

    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    log.info("Running %d questions against %s", len(eval_set), args.app_url)

    results = []
    by_type: dict[str, list[bool]] = {}

    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        log.info("[%d/%d] %s", i, len(eval_set), q)
        try:
            actual_answer, sources = ask(args.app_url, q)
            source_titles = [s.get("title", "") for s in sources]
            result = {**item, "actual_answer": actual_answer, "sources_returned": source_titles}
            results.append(result)
            log.info("  Answer: %s", actual_answer[:120])
        except Exception as e:
            log.warning("  FAILED: %s", e)
            results.append({**item, "actual_answer": f"ERROR: {e}", "sources_returned": []})
        if i < len(eval_set):
            time.sleep(1)  # be polite to Render free tier

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Print summary table
    print("\n" + "="*80)
    print(f"EVAL RESULTS — {len(results)} questions")
    print("="*80)
    for r in results:
        dtype = r.get("data_type", "?")
        print(f"\n[{dtype.upper()}] Q: {r['question']}")
        print(f"  Expected: {r['expected_answer'][:120]}")
        print(f"  Actual:   {r['actual_answer'][:120]}")
        top_source = r['sources_returned'][0] if r['sources_returned'] else "none"
        print(f"  Top source: {top_source}")

    print(f"\nFull results saved to {RESULTS_PATH}")
    print("Review and manually mark pass/fail in eval_results.json")


if __name__ == "__main__":
    main()
