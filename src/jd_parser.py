import re
from pathlib import Path


def extract_text_from_docx(path: str) -> str:
    import zipfile
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as z:
        xml_content = z.read("word/document.xml")
    root = ET.fromstring(xml_content)
    texts = []
    for t in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
        if t.text:
            texts.append(t.text)
    raw = "".join(texts)
    raw = raw.replace("\u2013", "-").replace("\u2014", "--").replace("\u2019", "'")
    raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2026", "...")
    raw = raw.replace("\u00a0", " ").replace("\u200b", "")
    return raw


MUST_HAVE_SKILLS = [
    "python",
    "embedding",
    "retrieval",
    "vector database", "vector db", "pinecone", "weaviate", "qdrant", "milvus",
    "faiss",
    "opensearch",
    "elasticsearch",
    "sentence-transformers", "sentence transformer",
    "bge", "e5",
    "rank", "rerank", "re-rank",
    "ndcg", "mrr", "map",
    "evaluation", "eval",
]

NICE_TO_HAVE_SKILLS = [
    "llm fine-tuning", "fine-tuning", "fine tuning",
    "lora", "qlora", "peft",
    "learning-to-rank", "learning to rank", "ltr",
    "xgboost",
    "hr-tech", "recruiting tech", "marketplace",
    "distributed systems",
    "open-source", "open source",
]

DISQUALIFIED_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "mumbai", "delhi", "gurgaon", "gurugram",
    "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata",
}

TIER_1_CITIES = {
    "pune", "noida", "mumbai", "delhi", "gurgaon", "gurugram",
    "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata",
    "ahmedabad",
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "lti", "mindtree",
}


def parse_jd(path: str) -> dict:
    raw = extract_text_from_docx(path)

    jd = {
        "raw_text": raw,
        "required_skills": MUST_HAVE_SKILLS,
        "nice_to_have_skills": NICE_TO_HAVE_SKILLS,
        "experience_range": (4, 10),
        "disqualified_companies": DISQUALIFIED_COMPANIES,
        "preferred_locations": PREFERRED_LOCATIONS,
        "tier_1_cities": TIER_1_CITIES,
        "consulting_firms": CONSULTING_FIRMS,
        "work_mode": "hybrid",
        "notice_preference_days": 30,
        "title_keywords": [
            "ai engineer", "machine learning engineer", "ml engineer",
            "nlp engineer", "senior ai", "senior machine learning",
            "applied scientist", "research engineer",
        ],
        "disqualifier_patterns": {
            "consulting_only": CONSULTING_FIRMS,
            "research_only": ["research", "scientist", "academic", "lab"],
            "cv_vision_only": ["computer vision", "cv", "image classification",
                                "object detection", "yolo", "speech",
                                "robotics"],
            "framework_enthusiast": ["langchain", "llamaindex"],
        },
    }

    exp_match = re.search(r"(\d+)\s*[-u2013u2014]\s*(\d+)\s*years", raw.lower())
    if exp_match:
        lo, hi = int(exp_match.group(1)), int(exp_match.group(2))
        jd["experience_range"] = (lo, hi)

    notice_match = re.search(r"(\d+)[+-]\s*day", raw.lower())
    if notice_match:
        jd["notice_preference_days"] = int(notice_match.group(1))

    return jd
