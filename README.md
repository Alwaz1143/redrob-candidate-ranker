# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A ranking system that scores 100,000 candidate profiles against a Senior AI Engineer job description and outputs the top 100 best-fit candidates. The system uses recruiter-quality reasoning — weighing skill depth, career history, production relevance, company background, and 23 behavioral signals — instead of naive keyword matching. Built for the Data & AI Challenge hackathon. [file:7][file:5]

**Important:** The full candidate dataset (`data/candidates.jsonl`) is excluded from this repo for size/distribution reasons, but the ranking code is fully reproducible once the dataset is placed at the expected path. [file:7]

---

## Live sandbox demo

A small-sample sandbox (50 candidates) is hosted as a Docker + Streamlit Space on Hugging Face. It runs the **same ranking pipeline** as the CLI and produces a ranked CSV for a tiny candidate set.

- Sandbox: `https://huggingface.co/spaces/alwaz/redrob-candidate-ranker-streamlit`  *(replace with your exact URL)* [file:7]

The sandbox is only meant for ≤100 candidates, as allowed by the hackathon spec; the full 100K run is reproduced via the CLI on CPU-only hardware. [file:7]

---

## Folder structure

```text
.
├── data/
│   ├── candidate_schema.json         # Full JSON Schema of a candidate record
│   ├── candidates.jsonl              # (NOT committed — place here manually)
│   ├── job_description.docx          # The JD to rank against
│   ├── README_bundle.docx            # Original hackathon README
│   ├── redrob_signals_doc.docx       # Behavioral signals reference
│   ├── sample_candidates.json        # 50 sample records for testing
│   ├── sample_submission.csv         # Format reference only
│   └── submission_spec.docx          # Output rules and constraints
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Streaming NDJSON/JSON loader
│   ├── jd_parser.py                  # JD text extraction and parsing
│   ├── feature_engineering.py        # Per-candidate feature computation
│   ├── scoring.py                    # Weighted scoring formula + reasoning
│   ├── pipeline.py                   # Shared rank_candidates() pipeline
│   └── rank.py                       # CLI entrypoint using pipeline.py
├── sandbox/
│   ├── app.py                        # Streamlit sandbox app (sample-only)
│   ├── requirements.txt              # Sandbox-specific deps
│   ├── sample_data.json              # 50-candidate subset for demo
│   └── job_description.docx          # JD copy for sandbox
├── tools/
│   └── validate_submission.py        # Official submission validator
├── requirements.txt                  # Core project dependencies
├── submission_metadata.yaml          # Mirrors portal metadata
└── submission_metadata_template.yaml # Original template from bundle
```

---

## Setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Dataset note

The full 100,000-candidate file (`data/candidates.jsonl`, ~487 MB) is **not included** in this repository due to its size and distribution restrictions. You must obtain it from the hackathon organizers and place it at: [file:7]

```text
data/candidates.jsonl
```

The sample files in `data/` (`sample_candidates.json`, `job_description.docx`, etc.) are committed and can be used for understanding the data shape and running quick tests. [file:7]

---

## Commands

All commands are run from the repository root. [file:7]

**Full ranking (100K candidates):**

```bash
python -m src.rank --candidates data/candidates.jsonl --jd data/job_description.docx --out team_xxx.csv
```

**Quick test (first 5 candidates from sample):**

```bash
python -m src.rank --candidates data/sample_candidates.json --jd data/job_description.docx --out team_xxx.csv --limit 5
```

**Validate output (spec checks: 100 rows, ranks 1–100, monotone scores, valid IDs):**

```bash
python tools/validate_submission.py team_xxx.csv
```

These commands satisfy the hackathon’s requirement for a single reproducible command to generate the submission CSV within 5 minutes on a 16 GB CPU-only machine. [file:7]

---

## Architecture summary

1. **JD parsing**  
   Extracts structured requirements from the `.docx` file: must-have skills (embeddings, vector DBs, evaluation frameworks), nice-to-haves (fine-tuning, LTR), disqualifiers (consulting-only careers, pure research), location/work-mode preferences, and notice-period expectations. [file:2][file:7]

2. **Candidate loading**  
   Streams `candidates.jsonl` (NDJSON) via a generator in `data_loader.py`, resilient to malformed lines and supporting both full runs and `--limit` testing. [file:3][file:7]

3. **Feature engineering**  
   Computes multiple feature families per candidate in `feature_engineering.py`: [file:5][file:7]

   - Skill match score (weighted by proficiency × duration × endorsements, with assessment-score bonuses).  
   - Role/seniority fit (title matching, experience band, applied-ML vs pure-research history).  
   - Company background (product vs. consulting firm detection, with consulting-only penalties).  
   - Location/work-mode fit (Pune/Noida/Tier-1 cities and relocation willingness).  
   - Behavioral composite from 23 platform signals (response rate, recency, open-to-work, etc.).  
   - Honeypot/consistency penalties (salary inversions, skill depth mismatches, title–description mismatches, education overlaps, timeline inconsistencies, generic summaries).  
   - TF‑IDF semantic similarity between candidate text and the JD (unigram, CPU-friendly).  

4. **Shared ranking pipeline (`pipeline.py`)**  
   Implements `rank_candidates(candidates, jd, max_n=100)` which ties everything together: building the TF‑IDF matrix, computing scores via `scoring.py`, applying monotonic score clamping, deterministic tie‑breaking, and generating reasoning strings. Both the CLI (`src/rank.py`) and the sandbox (`sandbox/app.py`) call this shared function, ensuring consistent behavior. [file:7]

5. **Weighted scoring**  
   Combines features via a formula of the form `base_score * behavioral_modifier - honeypot_penalty`, with tunable weights stored in `scoring.py`. This emphasizes must‑have AI skills, production relevance, and good behavioral signals while pushing inconsistent or honeypot profiles down the ranking. [file:5][file:7]

6. **Reasoning generation**  
   Produces 1–2 sentence, template-based justifications per candidate (no LLM calls), referencing concrete facts (years of experience, skills, location, notice period, and any concerns) to align with the hackathon’s reasoning quality criteria. [file:7]

---

## Sandbox / demo

The sandbox demo runs the same `rank_candidates` pipeline on a preloaded 50‑candidate sample and lets you:

- Run the ranking,
- Inspect the ranked table (candidate_id, rank, score, reasoning),
- Download a CSV of the small-sample output.

It is implemented in `sandbox/app.py` and deployed as a Docker + Streamlit Space on Hugging Face (see link above). The hackathon spec only requires the sandbox to handle ≤100 candidates; full 100K reproduction happens via the CLI command. [file:7]

---

If you share the exact Space URL, it can be inserted directly into the README and `submission_metadata.yaml` so everything matches perfectly.