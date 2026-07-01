WEIGHTS = {
    "skill_match": 0.30,
    "role_seniority": 0.25,
    "company_background": 0.10,
    "location_fit": 0.05,
    "behavioral": 0.15,
    "semantic": 0.15,
}

HONEYPOT_PENALTY_MULTIPLIER = 0.25


def compute_final_score(features: dict) -> float:
    skill = features.get("skill_match", {})
    role = features.get("role_seniority", {})
    company = features.get("company_background", {})
    location = features.get("location_fit", {})
    behavioral = features.get("behavioral", {})
    honeypot = features.get("honeypot_penalties", {})
    semantic = features.get("semantic_similarity", 0.0)

    req_match = skill.get("required_match_weighted", 0.0)
    has_assessment = skill.get("assessment_bonus", 0.0)
    nice_count = skill.get("nice_match_count", 0)
    shallow_penalty = skill.get("shallow_skill_count", 0) * 0.05

    skill_score = min(1.0, req_match * 0.5 + has_assessment * 0.3 + min(nice_count * 0.05, 0.2)) - shallow_penalty
    skill_score = max(0.0, skill_score)

    jd_title = role.get("jd_title_match", 0.0)
    exp_fit = role.get("exp_fit", 0.0)
    ml_roles = role.get("has_ml_roles", 0.0)
    role_score = jd_title * 0.5 + exp_fit * 0.25 + ml_roles * 0.25

    is_consulting = company.get("all_consulting", False)
    has_product = company.get("has_product_company", False)
    if is_consulting:
        company_score = 0.1
    else:
        company_score = 0.5
        if has_product:
            company_score = 1.0
    company_score = min(1.0, company_score)

    is_india = location.get("is_india", False)
    in_tier1 = location.get("in_tier1_city", False)
    willing = location.get("willing_relocate", False)

    if in_tier1:
        location_score = 1.0
    elif is_india:
        location_score = 0.6
    elif willing:
        location_score = 0.4
    else:
        location_score = 0.1

    behavioral_score = behavioral.get("composite", 0.5)

    base = (
        WEIGHTS["skill_match"] * skill_score
        + WEIGHTS["role_seniority"] * role_score
        + WEIGHTS["company_background"] * company_score
        + WEIGHTS["location_fit"] * location_score
        + WEIGHTS["behavioral"] * behavioral_score
        + WEIGHTS["semantic"] * semantic
    )

    hp_total = honeypot.get("total_penalty", 0.0)
    hp_mult = 1.0 - HONEYPOT_PENALTY_MULTIPLIER * hp_total
    hp_mult = max(0.0, hp_mult)

    final = base * hp_mult

    return round(final, 6)


def generate_reasoning(candidate: dict, features: dict, score: float, rank: int) -> str:
    p = candidate.get("profile", {})
    skill = features.get("skill_match", {})
    role = features.get("role_seniority", {})
    behavioral = features.get("behavioral", {})
    honeypot = features.get("honeypot_penalties", {})
    semantic = features.get("semantic_similarity", 0.0)

    title = p.get("current_title", "Professional")
    yoe = p.get("years_of_experience", 0)
    loc = p.get("location", "")
    country = p.get("country", "")

    parts = []

    parts.append(f"{title} with {yoe}yr exp")

    req_count = skill.get("required_match_count", 0)
    if req_count > 0:
        parts.append(f"{req_count} core skills matched")

    ml_roles = role.get("has_ml_roles", 0.0)
    if ml_roles >= 0.8:
        parts.append("strong ML/AI career history")
    elif ml_roles >= 0.3:
        parts.append("some ML/AI exposure")

    response_rate = behavioral.get("response_rate", 0.0)
    if response_rate >= 0.5:
        parts.append(f"response rate {response_rate:.2f}")

    open_to_work = behavioral.get("open_to_work", False)
    if not open_to_work:
        parts.append("not open to work")

    hp_total = honeypot.get("total_penalty", 0.0)
    if hp_total > 0.5:
        parts.append("⚠ profile inconsistencies detected")

    if country == "India" and loc:
        parts.append(f"based in {loc}")
    elif "India" in str(country):
        parts.append("India-based")

    semantic_pct = int(semantic * 100)
    if semantic_pct > 20:
        parts.append(f"JD relevance {semantic_pct}%")

    reasoning = "; ".join(parts)
    return reasoning[:250]
