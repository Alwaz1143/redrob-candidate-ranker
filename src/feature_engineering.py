import re
import math
from datetime import datetime, date

REFERENCE_DATE = date(2026, 7, 1)

PROFICIENCY_MAP = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}

JD_TITLE_KEYWORDS = [
    "ai engineer", "machine learning engineer", "ml engineer",
    "nlp engineer", "applied scientist", "research engineer",
    "data scientist", "deep learning",
]

JD_SENIORITY_KEYWORDS = ["senior", "lead", "staff", "principal", "founding", "foundation"]

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "lti", "mindtree",
}

PRODUCT_FIRMS_BONUS = {
    "google", "meta", "microsoft", "amazon", "apple", "netflix",
    "uber", "lyft", "airbnb", "stripe", "shopify", "spotify",
    "linkedin", "twitter", "salesforce", "oracle", "adobe",
    "reddit", "pinterest", "snap", "doordash", "robinhood",
}


def get_text_for_tfidf(candidate: dict) -> str:
    p = candidate.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("description", ""))
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
    return " ".join(parts)


def compute_skill_match(candidate: dict, jd: dict) -> dict:
    skills = candidate.get("skills", [])
    required = set(s.lower() for s in jd["required_skills"])
    nice = set(s.lower() for s in jd["nice_to_have_skills"])
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    required_match_count = 0
    required_match_weighted = 0.0
    nice_match_count = 0
    shallow_skill_count = 0
    total_skill_depth = 0.0
    assessment_bonus = 0.0

    for skill in skills:
        name = skill.get("name", "").lower().strip()
        prof = PROFICIENCY_MAP.get(skill.get("proficiency", ""), 0.0)
        dur = skill.get("duration_months", 0) or 0
        end = skill.get("endorsements", 0) or 0

        depth = prof * min(1.0, dur / 24.0) * min(1.0, 0.2 + 0.8 * min(1.0, end / 10.0))
        total_skill_depth += depth

        is_shallow = (prof >= 0.75 and dur < 3) or (prof >= 0.75 and end == 0)
        if is_shallow:
            shallow_skill_count += 1

        matched_req = False
        for req in required:
            if req in name or name in req:
                matched_req = True
                break
        if matched_req:
            required_match_count += 1
            required_match_weighted += depth

        matched_nice = False
        for n in nice:
            if n in name or name in n:
                matched_nice = True
                break
        if matched_nice:
            nice_match_count += 1

        if name in assessments:
            assessment_bonus += assessments[name] / 100.0

    assessment_bonus = min(assessment_bonus, 1.0)

    return {
        "required_match_count": required_match_count,
        "required_match_weighted": required_match_weighted,
        "nice_match_count": nice_match_count,
        "shallow_skill_count": shallow_skill_count,
        "total_skill_depth": total_skill_depth,
        "assessment_bonus": assessment_bonus,
    }


def compute_role_seniority(candidate: dict, jd: dict) -> dict:
    p = candidate.get("profile", {})
    yoe = p.get("years_of_experience", 0) or 0
    exp_lo, exp_hi = jd["experience_range"]
    exp_center = (exp_lo + exp_hi) / 2.0

    title_text = (p.get("current_title", "") + " " + p.get("headline", "")).lower()
    all_titles = [p.get("current_title", "").lower()]
    for role in candidate.get("career_history", []):
        all_titles.append(role.get("title", "").lower())
    all_title_text = " ".join(all_titles)

    seniority_score = 0.0
    if any(kw in all_title_text for kw in JD_SENIORITY_KEYWORDS):
        seniority_score += 0.3

    jd_title_match = 0.0
    for kw in JD_TITLE_KEYWORDS:
        if kw in all_title_text:
            jd_title_match = max(jd_title_match, 0.7)
            if any(k in title_text for k in JD_SENIORITY_KEYWORDS):
                jd_title_match = max(jd_title_match, 1.0)

    exp_fit = 0.0
    if yoe >= exp_lo - 1 and yoe <= exp_hi + 2:
        exp_fit = 1.0 - abs(yoe - exp_center) / (exp_hi - exp_lo + 2)
        exp_fit = max(0.0, exp_fit)
    else:
        exp_fit = 0.1

    has_ml_roles = 0.0
    for role in candidate.get("career_history", []):
        desc = (role.get("title", "") + " " + role.get("description", "")).lower()
        ml_kw = ["machine learning", "deep learning", "nlp", "ai engineer",
                 "data science", "recommendation", "ranking", "embedding",
                 "llm", "fine-tun", "pytorch", "tensorflow"]
        if any(k in desc for k in ml_kw):
            has_ml_roles = max(has_ml_roles, 0.8)
            if any(k in desc for k in ["senior", "lead"]):
                has_ml_roles = max(has_ml_roles, 1.0)

    return {
        "yoe": yoe,
        "exp_fit": exp_fit,
        "jd_title_match": jd_title_match,
        "seniority_score": seniority_score,
        "has_ml_roles": has_ml_roles,
    }


def compute_company_background(candidate: dict, jd: dict) -> dict:
    p = candidate.get("profile", {})
    all_companies = [p.get("current_company", "")]
    for role in candidate.get("career_history", []):
        all_companies.append(role.get("company", ""))

    company_names_lower = [c.lower().strip() for c in all_companies if c]
    current_company = p.get("current_company", "").lower().strip()

    has_product = False
    for c in company_names_lower:
        for bonus_firm in PRODUCT_FIRMS_BONUS:
            if bonus_firm in c or c in bonus_firm:
                has_product = True
                break

    all_consulting = all(
        any(consult in c for consult in CONSULTING_FIRMS)
        for c in company_names_lower
    ) if company_names_lower else False

    current_is_consulting = any(
        consult in current_company for consult in CONSULTING_FIRMS
    ) if current_company else False

    industry = p.get("current_industry", "").lower()
    ai_industry = any(kw in industry for kw in ["ai", "ml", "software", "technology", "internet"])

    return {
        "all_consulting": all_consulting,
        "current_is_consulting": current_is_consulting,
        "has_product_company": has_product,
        "ai_industry": ai_industry,
        "company_count": len(company_names_lower),
    }


def compute_location_fit(candidate: dict, jd: dict) -> dict:
    p = candidate.get("profile", {})
    loc = (p.get("location", "") + " " + p.get("country", "")).lower()
    signals = candidate.get("redrob_signals", {})

    is_india = "india" in loc
    in_tier1 = any(city in loc for city in jd["tier_1_cities"])
    in_preferred = any(city in loc for city in jd["preferred_locations"])
    willing_relocate = signals.get("willing_to_relocate", False)

    pref_work_mode = signals.get("preferred_work_mode", "").lower()
    jd_mode = jd.get("work_mode", "hybrid")

    return {
        "is_india": is_india,
        "in_tier1_city": in_tier1 or in_preferred,
        "willing_relocate": willing_relocate,
        "preferred_work_mode": pref_work_mode,
        "mode_fits_hybrid": pref_work_mode in ("hybrid", "flexible", "remote"),
    }


def compute_behavioral_score(candidate: dict) -> dict:
    s = candidate.get("redrob_signals", {})
    now = REFERENCE_DATE

    response_rate = s.get("recruiter_response_rate", 0.0) or 0.0
    open_to_work = s.get("open_to_work_flag", False)
    interview_rate = s.get("interview_completion_rate", 0.0) or 0.0
    github = s.get("github_activity_score", -1) or -1
    completeness = s.get("profile_completeness_score", 0) or 0
    saved = s.get("saved_by_recruiters_30d", 0) or 0
    views = s.get("profile_views_received_30d", 0) or 0
    search_appearances = s.get("search_appearance_30d", 0) or 0
    verified_email = s.get("verified_email", False)
    verified_phone = s.get("verified_phone", False)

    last_active_str = s.get("last_active_date", "")
    days_since_active = 999
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_since_active = (now - last_active).days
        except (ValueError, TypeError):
            days_since_active = 999

    recency_score = 1.0
    if days_since_active <= 7:
        recency_score = 1.0
    elif days_since_active <= 30:
        recency_score = 0.8
    elif days_since_active <= 90:
        recency_score = 0.5
    elif days_since_active <= 180:
        recency_score = 0.3
    else:
        recency_score = 0.1

    response_modifier = 0.3 + 0.7 * response_rate

    github_score = 0.0
    if github >= 0:
        github_score = github / 100.0

    composite = (
        0.20 * (1.0 if open_to_work else 0.3)
        + 0.25 * response_modifier
        + 0.10 * recency_score
        + 0.10 * interview_rate
        + 0.10 * github_score
        + 0.05 * min(1.0, completeness / 100.0)
        + 0.05 * min(1.0, saved / 10.0)
        + 0.05 * min(1.0, views / 50.0)
        + 0.05 * (1.0 if verified_email else 0.3)
        + 0.05 * (1.0 if verified_phone else 0.3)
    )

    return {
        "composite": composite,
        "response_rate": response_rate,
        "open_to_work": open_to_work,
        "recency_score": recency_score,
        "days_since_active": days_since_active,
        "github_score": github_score,
        "completeness": completeness / 100.0 if completeness else 0,
    }


def compute_honeypot_penalties(candidate: dict) -> dict:
    penalties = {}
    p = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    salary = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary.get("min", 0) or 0
    sal_max = salary.get("max", 0) or 0
    if sal_min > sal_max and sal_max > 0:
        penalties["salary_inversion"] = 1
    else:
        penalties["salary_inversion"] = 0

    shallow_count = 0
    for skill in candidate.get("skills", []):
        prof = PROFICIENCY_MAP.get(skill.get("proficiency", ""), 0.0)
        dur = skill.get("duration_months", 0) or 0
        end = skill.get("endorsements", 0) or 0
        if prof >= 0.75 and (dur < 3 or end == 0):
            shallow_count += 1
    penalties["skill_depth_mismatch"] = min(shallow_count, 10) / 10.0

    yoe = p.get("years_of_experience", 0) or 0
    total_career_months = sum(
        (role.get("duration_months", 0) or 0)
        for role in candidate.get("career_history", [])
    )
    career_years = total_career_months / 12.0
    if yoe > 0 and career_years > 0:
        ratio = abs(yoe - career_years) / max(yoe, career_years)
        if ratio > 0.5 and (yoe > 2 or career_years > 2):
            penalties["exp_timeline_mismatch"] = min(ratio, 1.0)
        else:
            penalties["exp_timeline_mismatch"] = 0.0
    else:
        penalties["exp_timeline_mismatch"] = 0.0

    edu_entries = candidate.get("education", [])
    edu_overlap = 0
    for i in range(len(edu_entries)):
        for j in range(i + 1, len(edu_entries)):
            e1_start = edu_entries[i].get("start_year", 0) or 0
            e1_end = edu_entries[i].get("end_year", 0) or 0
            e2_start = edu_entries[j].get("start_year", 0) or 0
            e2_end = edu_entries[j].get("end_year", 0) or 0
            if e1_start >= e2_start and e1_start <= e2_end:
                edu_overlap += 1
            elif e2_start >= e1_start and e2_start <= e1_end:
                edu_overlap += 1
    penalties["education_overlap"] = min(edu_overlap, 5) / 5.0

    summary = p.get("summary", "").lower()
    title = p.get("current_title", "").lower()
    generic_phrases = [
        "i've spent my career in marketing manager",
        "i'm a marketing manager",
        "lately i've been curious about how ai tools",
        "i've experimented with chatgpt",
        "open to roles where i can apply my domain expertise",
        "driving business outcomes",
        "helping teams scale",
    ]
    generic_count = sum(1 for phrase in generic_phrases if phrase in summary)
    penalties["generic_summary"] = min(generic_count, 5) / 5.0

    title_desc_mismatch = 0
    for role in candidate.get("career_history", []):
        rtitle = role.get("title", "").lower()
        desc = role.get("description", "").lower()
        mismatch_pairs = [
            ("accountant", "mechanical engineer"),
            ("civil engineer", "brand design"),
            ("mechanical engineer", "enterprise sales"),
            ("operations manager", "accounting role"),
            ("marketing manager", "customer support team lead"),
            ("hr manager", "accounting role"),
            ("business analyst", "mechanical engineer"),
            ("content writer", "mechanical engineer"),
            ("accountant", "sales"),
            ("frontend engineer", "test automation"),
        ]
        for t1, t2 in mismatch_pairs:
            if t1 in rtitle and t2 in desc:
                title_desc_mismatch += 1
            if t2 in rtitle and t1 in desc:
                title_desc_mismatch += 1
    penalties["title_desc_mismatch"] = min(title_desc_mismatch, 5) / 5.0

    total = sum(penalties.values())
    penalties["total_penalty"] = total

    return penalties


def compute_semantic_similarity(tfidf_matrix, idx: int) -> float:
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix[-1:])[0][0]
    return float(sim)
