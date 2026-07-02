#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rank")

from src.data_loader import stream_candidates, detect_format
from src.jd_parser import parse_jd
from src.pipeline import rank_candidates


def main():
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Intelligent Candidate Discovery & Ranking"
    )
    parser.add_argument(
        "--candidates", "-c",
        default="data/candidates.jsonl",
        help="Path to candidates file (jsonl/json/jsonl.gz)",
    )
    parser.add_argument(
        "--jd", "-j",
        default="data/job_description.docx",
        help="Path to job description docx",
    )
    parser.add_argument(
        "--out", "-o",
        default="team_xxx.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Process only first N candidates (for testing)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write per-candidate feature debug JSON",
    )
    args = parser.parse_args()

    fmt = detect_format(args.candidates)
    logger.info(f"Candidates file format: {fmt}")
    logger.info(f"Parsing JD: {args.jd}")

    t0 = time.time()
    jd = parse_jd(args.jd)
    logger.info(f"JD parsed in {time.time()-t0:.1f}s")
    logger.info(f"Experience range: {jd['experience_range']}")
    logger.info(f"Required skills: {len(jd['required_skills'])} keywords")

    logger.info(f"Streaming candidates from: {args.candidates}")
    t1 = time.time()

    all_candidates = []
    for i, c in enumerate(stream_candidates(args.candidates)):
        if args.limit and i >= args.limit:
            break
        all_candidates.append(c)
        if i > 0 and i % 25000 == 0:
            logger.info(f"  Loaded {i} candidates...")

    load_time = time.time() - t1
    logger.info(f"Loaded {len(all_candidates)} candidates in {load_time:.1f}s")

    logger.info("Running ranking pipeline...")
    t3 = time.time()

    rows = rank_candidates(all_candidates, jd, max_n=100)

    score_time = time.time() - t3
    logger.info(f"Ranking completed in {score_time:.1f}s")

    out_path = args.out
    logger.info(f"Writing submission to: {out_path}")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['score']:.4f}",
                r["reasoning"],
            ])

    total_time = time.time() - t0
    logger.info(f"Done! Total time: {total_time:.1f}s")
    logger.info(f"Output: {out_path}")

    if args.debug:
        debug_path = out_path.replace(".csv", "_features.json")
        debug_data = []
        for r in rows[:20]:
            debug_data.append({
                "rank": r["rank"],
                "candidate_id": r["candidate_id"],
                "score": r["score"],
            })
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, indent=2, default=str)
        logger.info(f"Debug features written to: {debug_path}")


if __name__ == "__main__":
    main()
