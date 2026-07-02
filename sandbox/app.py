import sys
import time
import csv
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from src.data_loader import stream_candidates
from src.jd_parser import parse_jd
from src.pipeline import rank_candidates


SAMPLE_PATH = Path(__file__).resolve().parent / "sample_data.json"
JD_PATH = Path(__file__).resolve().parent / "job_description.docx"

st.set_page_config(page_title="Redrob Candidate Ranker", layout="centered")
st.title("Redrob Hackathon — Intelligent Candidate Discovery & Ranking")
st.markdown(
    "Demo sandbox that ranks candidates against a Senior AI Engineer job "
    "description. Runs the same scoring pipeline as the CLI — CPU-only, "
    "no external API calls."
)

if "candidates" not in st.session_state:
    candidates = list(stream_candidates(str(SAMPLE_PATH)))
    st.session_state.candidates = candidates
    st.session_state.candidate_count = len(candidates)

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.runtime = None

jd_source = st.radio(
    "Job Description source",
    ["Use bundled JD (sandbox/job_description.docx)", "Upload a different JD (.docx)"],
    horizontal=True,
)

uploaded_jd = None
if "Upload" in jd_source:
    uploaded_jd = st.file_uploader("Upload JD", type=["docx"])

if st.button("Run Ranking on Sample", type="primary", use_container_width=True):
    with st.spinner("Ranking candidates..."):
        t0 = time.time()

        if uploaded_jd:
            tmp_path = Path(__file__).resolve().parent / "_tmp_upload.docx"
            tmp_path.write_bytes(uploaded_jd.getvalue())
            jd = parse_jd(str(tmp_path))
            tmp_path.unlink(missing_ok=True)
        else:
            jd = parse_jd(str(JD_PATH))

        candidates = st.session_state.candidates
        rows = rank_candidates(candidates, jd, max_n=100)

        runtime = time.time() - t0
        st.session_state.results = rows
        st.session_state.runtime = runtime

if st.session_state.results is not None:
    rows = st.session_state.results
    runtime = st.session_state.runtime

    st.subheader("Ranked Results")
    col1, col2 = st.columns(2)
    col1.metric("Candidates Ranked", len(rows))
    col2.metric("Runtime", f"{runtime:.2f}s")

    df = pd.DataFrame(rows)
    df["score"] = df["score"].apply(lambda x: f"{x:.4f}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in rows:
        writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", r["reasoning"]])
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="sandbox_ranking.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info(f"Sample pool loaded: {st.session_state.candidate_count} candidates. Click the button to rank.")
