# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A ranking system that scores 100,000 candidate profiles against a Senior AI Engineer job description and outputs the top 100 best-fit candidates. The system uses recruiter-quality reasoning — weighing skill depth, career history, production relevance, company background, and 23 behavioral signals — instead of naive keyword matching. Built for the Data & AI Challenge hackathon.

**Important:** The full candidate dataset (`data/candidates.jsonl`) is excluded from this repo for size/distribution reasons, but the ranking code is fully reproducible once the dataset is placed at the expected path.

## Folder Structure

```
.
├── data/
│   ├── candidate_schema.json        # Full JSON Schema of a candidate record
│   ├── candidates.jsonl             # (NOT committed — place here manually)
│   ├── job_description.docx         # The JD to rank against
│   ├── README_bundle.docx           # Original hackathon README
│   ├── redrob_signals_doc.docx      # Behavioral signals reference
│   ├── sample_candidates.json       # 50 sample records for testing
│   ├── sample_submission.csv        # Format reference only
│   └── submission_spec.docx         # Output rules and constraints
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Streaming NDJSON/JSON loader
│   ├── feature_engineering.py       # Per-candidate feature computation
│   ├── jd_parser.py                 # JD text extraction and parsing
│   ├── rank.py                      # CLI entrypoint
│   └── scoring.py                   # Weighted scoring formula + reasoning
├── tools/
│   └── validate_submission.py       # Official submission validator
├── requirements.txt
├── submission_metadata.yaml
└── submission_metadata_template.yaml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

## Dataset Note

The full 100,000-candidate file (`data/candidates.jsonl`, ~487 MB) is **not included** in this repository due to its size and distribution restrictions. You must obtain it from the hackathon organizers and place it at:

```
data/candidates.jsonl
```

The sample files in `data/` (`sample_candidates.json`, `job_description.docx`, etc.) are committed and can be used for understanding the data shape and running quick tests.

## Commands

All commands are run from the repository root.

**Full ranking (100K candidates):**
```bash
python -m src.rank --candidates data/candidates.jsonl --jd data/job_description.docx --out team_xxx.csv
```

**Quick test (first 5 candidates from sample):**
```bash
python -m src.rank --candidates data/sample_candidates.json --jd data/job_description.docx --out team_xxx.csv --limit 5
```

**Validate output:**
```bash
python tools/validate_submission.py team_xxx.csv
```

## Constraints

- CPU only — no GPU used at any point
- Zero network calls during ranking — all computation is local
- Observed full run time: ~100 seconds (well under the 5-minute limit)
- Peak memory: ~3 GB for 100K candidates + TF-IDF matrix

## Architecture Summary

1. **JD Parsing** — Extracts structured requirements from the `.docx` file: must-have skills (embeddings, vector DBs, eval frameworks), nice-to-haves (fine-tuning, LTR), disqualifiers (consulting-only backgrounds), location/work-mode preferences, and notice-period expectations.

2. **Candidate Loading** — Streams `candidates.jsonl` (NDJSON) via a generator; handles malformed lines gracefully.

3. **Feature Engineering** — Computes 7 feature families per candidate:
   - Skill match score (weighted by proficiency × duration × endorsements, with assessment-score bonus)
   - Role/seniority fit (title matching, experience band, ML career history depth)
   - Company background (product vs. consulting firm detection)
   - Location/work-mode fit (Tier-1 Indian cities, relocation willingness)
   - Behavioral composite (response rate, activity recency, open-to-work, GitHub activity, interview rate, profile completeness)
   - Honeypot/consistency penalties (salary inversions, skill depth mismatches, title-desc mismatches, education overlaps, timeline inconsistencies, generic summaries)
   - TF-IDF semantic similarity between candidate text and the JD (unigram, CPU-friendly)

4. **Weighted Scoring** — Combines features via `base_score * behavioral_modifier - honeypot_penalty` with tunable weights in `src/scoring.py`.

5. **Reasoning Generation** — Template-based 1-2 sentence justifications from computed features (no LLM calls).

## Next Steps

A sandbox/demo app (Streamlit or similar) will be added later. This CLI is the first deliverable.
