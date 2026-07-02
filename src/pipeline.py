import logging

logger = logging.getLogger(__name__)

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
from src.scoring import compute_final_score, generate_reasoning


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


def rank_candidates(candidates, jd, max_n=100):
    candidate_texts = [get_text_for_tfidf(c) for c in candidates]
    tfidf_matrix, _ = build_tfidf_matrix(candidate_texts, jd["raw_text"])

    scored = []
    for idx, c in enumerate(candidates):
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

    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))

    top_n = min(len(scored), max_n)
    ranked = scored[:top_n]

    prev_score = float("inf")
    for i in range(len(ranked)):
        score, c, features = ranked[i]
        if score > prev_score:
            score = prev_score
        ranked[i] = (score, c, features)
        prev_score = score

    groups = {}
    for entry in ranked:
        display_s = f"{entry[0]:.4f}"
        groups.setdefault(display_s, []).append(entry)

    final_ranked = []
    for display_s in sorted(groups.keys(), reverse=True, key=float):
        group = groups[display_s]
        group.sort(key=lambda x: x[1]["candidate_id"])
        for score, c, features in group:
            final_ranked.append((len(final_ranked) + 1, score, c, features))

    rows = []
    for rank, score, c, features in final_ranked:
        reasoning = generate_reasoning(c, features, score, rank)
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": rank,
            "score": score,
            "reasoning": reasoning,
        })

    return rows
