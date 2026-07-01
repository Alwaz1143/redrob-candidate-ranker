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
from src.feature_engineering import (
    get_text_for_tfidf,
    compute_skill_match,
    compute_role_seniority,
    compute_company_background,
    compute_location_fit,
    compute_behavioral_score,
    compute_honeypot_penalties,
    compute_semantic_similarity,
)
from src.scoring import compute_final_score, generate_reasoning, WEIGHTS


def build_tfidf_matrix(docs, jd_text, limit=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = []
    for i, doc in enumerate(docs):
        if limit and i >= limit:
            break
        texts.append(doc)
    jd_words = " ".join(jd_text.split()[:500])
    texts.append(jd_words)
    vec = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        ngram_range=(1, 1),
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(texts)
    logger.info(f"TF-IDF matrix built: {matrix.shape}")
    return matrix, vec


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

    logger.info("Building TF-IDF vectors (for semantic similarity)...")
    t2 = time.time()
    candidate_texts = [get_text_for_tfidf(c) for c in all_candidates]
    tfidf_matrix, _ = build_tfidf_matrix(candidate_texts, jd["raw_text"], limit=None)
    tfidf_time = time.time() - t2
    logger.info(f"TF-IDF built in {tfidf_time:.1f}s")

    logger.info("Computing features and scores for all candidates...")
    t3 = time.time()

    scored = []
    for idx, c in enumerate(all_candidates):
        skill_match = compute_skill_match(c, jd)
        role_sen = compute_role_seniority(c, jd)
        company = compute_company_background(c, jd)
        location = compute_location_fit(c, jd)
        behavioral = compute_behavioral_score(c)
        honeypot = compute_honeypot_penalties(c)
        semantic = compute_semantic_similarity(tfidf_matrix, idx)

        features = {
            "skill_match": skill_match,
            "role_seniority": role_sen,
            "company_background": company,
            "location_fit": location,
            "behavioral": behavioral,
            "honeypot_penalties": honeypot,
            "semantic_similarity": semantic,
        }

        score = compute_final_score(features)
        scored.append((score, c, features))

        if idx > 0 and idx % 25000 == 0:
            logger.info(f"  Scored {idx} candidates...")

    score_time = time.time() - t3
    logger.info(f"Scoring completed in {score_time:.1f}s")

    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))

    top100 = scored[:100]

    prev_score = float("inf")
    for i in range(len(top100)):
        score, c, features = top100[i]
        if score > prev_score:
            score = prev_score
        top100[i] = (score, c, features)
        prev_score = score

    groups = {}
    for entry in top100:
        display_s = f"{entry[0]:.4f}"
        groups.setdefault(display_s, []).append(entry)

    final_ranked = []
    for display_s in sorted(groups.keys(), reverse=True, key=float):
        group = groups[display_s]
        group.sort(key=lambda x: x[1]["candidate_id"])
        for score, c, features in group:
            final_ranked.append((len(final_ranked) + 1, score, c, features))

    top100 = final_ranked

    out_path = args.out
    logger.info(f"Writing submission to: {out_path}")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, score, c, features in top100:
            reasoning = generate_reasoning(c, features, score, rank)
            writer.writerow([
                c["candidate_id"],
                rank,
                f"{score:.4f}",
                reasoning,
            ])

    total_time = time.time() - t0
    logger.info(f"Done! Total time: {total_time:.1f}s")
    logger.info(f"Output: {out_path}")

    if args.debug:
        debug_path = out_path.replace(".csv", "_features.json")
        debug_data = []
        for rank, score, c, features in top100[:20]:
            entry = {
                "rank": rank,
                "candidate_id": c["candidate_id"],
                "score": score,
                "features": {
                    k: v for k, v in features.items()
                    if k != "candidate"
                },
            }
            debug_data.append(entry)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, indent=2, default=str)
        logger.info(f"Debug features written to: {debug_path}")


if __name__ == "__main__":
    main()
