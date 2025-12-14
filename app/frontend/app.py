import base64
import json
import os
import time
from typing import Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
SUBMIT_ENDPOINT = os.getenv("SUBMIT_ENDPOINT", "/api/form/submit")
RESULT_ENDPOINT = os.getenv("RESULT_ENDPOINT", "/api/form/result")
MAX_CHARS = 2000
MAX_FILE_MB = 10


@st.cache_data(show_spinner=False)
def get_request_types() -> List[str]:
    return [
        "Price Strategy",
        "Growth & Marketing",
        "Hiring & Operations",
        "Customer & Revenue",
        "Strategic Decisions",
        "Other",
    ]


def encode_file(uploaded_file, max_mb: int = MAX_FILE_MB) -> Dict[str, str]:
    """Base64-encode an UploadedFile in chunks, enforcing a max size."""
    uploaded_file.seek(0)
    total = 0
    chunks: List[str] = []
    while True:
        chunk = uploaded_file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_mb * 1024 * 1024:
            raise ValueError(f"File '{uploaded_file.name}' exceeds {max_mb} MB limit")
        chunks.append(base64.b64encode(chunk).decode("utf-8"))

    return {
        "filename": uploaded_file.name,
        "mime_type": uploaded_file.type,
        "size_bytes": total,
        "data_b64": "".join(chunks),
    }


def poll_for_result(request_id: int, placeholder) -> None:
    """Poll backend for the final result and display formatted output."""
    status_area = placeholder.empty()
    for attempt in range(1, 13):  # Poll up to ~120s
        try:
            resp = requests.get(f"{BACKEND_URL}{RESULT_ENDPOINT}/{request_id}", timeout=10)
            if resp.status_code == 200:
                payload = resp.json()
                status = payload.get("status", "unknown")

                if status == "pending":
                    status_area.info(f"Processing... (attempt {attempt}/12)")
                    time.sleep(10)
                    continue

                # Clear status area
                status_area.empty()

                # Parse and display result
                result_str = payload.get("result", "{}")
                try:
                    result = json.loads(result_str) if isinstance(result_str, str) else result_str
                except Exception:
                    result = {"error": "Failed to parse result"}

                # Display formatted result
                if status == "done":
                    st.success("Analysis Complete!")

                    response = result.get("response", {})

                    # Display summary
                    st.subheader("Summary")
                    summary = response.get("summary", "No summary available")
                    confidence = response.get("confidence", 0)
                    st.write(summary)
                    st.metric("Confidence Score", f"{confidence:.0%}")

                    # Display insights
                    insights = response.get("insights", [])
                    if insights:
                        st.subheader("Key Insights")
                        for insight in insights:
                            st.info(insight)

                    # Display risks/limitations
                    risks = response.get("risks", [])
                    if risks:
                        with st.expander("Risks & Limitations"):
                            for risk in risks:
                                st.warning(risk)

                    # Display next steps
                    next_steps = response.get("next_steps", [])
                    if next_steps:
                        st.subheader("Recommended Next Steps")
                        for i, step in enumerate(next_steps, 1):
                            st.write(f"{i}. {step}")

                    # Disclaimers
                    disclaimers = response.get("disclaimers", [])
                    if disclaimers:
                        with st.expander("Disclaimers"):
                            for disclaimer in disclaimers:
                                st.caption(disclaimer)

                    # Show raw data in expander
                    with st.expander("View Technical Details"):
                        st.json(result)

                elif status == "error":
                    validation = result.get("validation", {})

                    # Validation rejection (early failure)
                    if validation and not validation.get("ok", True):
                        st.error("Data Validation Failed")
                        st.write("**Why this happened:**")
                        st.write(validation.get("message", "Unknown validation error"))

                        suggestions = validation.get("suggestions", [])
                        if suggestions:
                            st.subheader("How to Fix This")
                            for suggestion in suggestions:
                                st.info(f"• {suggestion}")

                        profiles = validation.get("profiles", [])
                        if profiles:
                            with st.expander("Data Summary"):
                                for profile in profiles:
                                    if "error" not in profile:
                                        st.write(f"**File:** {profile.get('file', 'Unknown')}")
                                        st.write(f"**Rows:** {profile.get('rows', 0)}")
                                        st.write(f"**Columns:** {', '.join(profile.get('columns', []))}")
                    else:
                        # Other error (execution/planning failure)
                        st.error("Analysis encountered an error")
                        response = result.get("response", {})
                        if response:
                            st.write("**Error Details:**")
                            st.write(response.get("summary", "Unknown error"))

                            validation_check = result.get("validation", {})
                            if not validation_check.get("ok", True):
                                st.warning("Data Validation Issues:")
                                for issue in validation_check.get("issues", []):
                                    st.write(f"- {issue}")
                                for suggestion in validation_check.get("suggestions", []):
                                    st.info(f"Suggestion: {suggestion}")

                    with st.expander("View Error Details"):
                        st.json(result)
                else:
                    st.warning(f"Unknown status: {status}")
                    st.json(payload)

                return

            elif resp.status_code == 404:
                status_area.info(f"Processing... (attempt {attempt}/12)")
            else:
                status_area.warning(f"Status {resp.status_code}: {resp.text}")

        except Exception as exc:
            status_area.warning(f"Polling error (attempt {attempt}/12): {exc}")

        time.sleep(10)

    status_area.error("Analysis is taking longer than expected. Please check back later or contact support.")


def main() -> None:
    st.set_page_config(page_title="Data Validator Agent", layout="centered")
    st.title("Data Validator Agent")
    st.write("Upload your CSV and ask a question. Files must be ≤ 10 MB and < 50,000 rows.")

    req_types = get_request_types()

    with st.form("analysis_form", clear_on_submit=False):
        request_type = st.selectbox("Request Type", req_types, index=0)

        question = st.text_area(
            "Your Question",
            max_chars=MAX_CHARS,
            help="Ask anything about your data. Example: 'What is the average revenue per region?'",
        )

        additional_context = st.text_area("Additional Context (Optional)")
        st.markdown("""
            <style>
            .uploadedFileSizeLimit {
                display: none;
            }
            </style>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="Upload CSV (≤ 10 MB)",
            type=["csv"],
            accept_multiple_files=False,
            help="Single CSV only. Size ≤ 10 MB. Rows < 50,000.",
        )

        submit_btn = st.form_submit_button("Analyze", use_container_width=True, type="primary")

    if submit_btn:
        errors = []
        if not request_type:
            errors.append("Request Type is required")
        if not question.strip():
            errors.append("Question is required")
        if uploaded_file is None:
            errors.append("A CSV file is required")
        else:
            size_mb = uploaded_file.size / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                errors.append("File must be ≤ 10 MB")

        if errors:
            st.error("\n".join(errors))
            return

        try:
            files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
            data = {
                "request_type": request_type,
                "question": question.strip(),
                "context": additional_context or None,
                "priority": "normal",
            }

            with st.spinner("Submitting for analysis..."):
                resp = requests.post(
                    f"{BACKEND_URL}{SUBMIT_ENDPOINT}",
                    data=data,
                    files=files,
                    timeout=300,
                )
        except Exception as exc:
            st.error(f"Failed to submit: {exc}")
            return

        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            st.error(f"Request failed: {resp.status_code} {detail}")
            return

        out = resp.json()
        if out.get("status") == "error":
            st.error(out.get("error", "Unknown error"))
            return

        request_id = out.get("id")
        st.success("Analysis started! Fetching results...")

        if request_id:
            poll_placeholder = st.empty()
            with st.spinner("Processing... This may take up to a minute"):
                poll_for_result(request_id, poll_placeholder)


if __name__ == "__main__":
    main()
