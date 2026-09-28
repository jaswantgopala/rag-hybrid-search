import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Hybrid Search", layout="wide")
st.title("RAG Hybrid Search Dashboard")

with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox("Chunking strategy", ["structure_aware", "fixed_size", "semantic"])
    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5)
    compare_mode = st.checkbox("Compare hybrid vs dense-only", value=False)

    st.divider()
    st.subheader("Indexed Documents")
    try:
        docs_response = requests.get(f"{API_URL}/v1/documents")
        docs_data = docs_response.json()
        for doc in docs_data.get("documents", []):
            st.text(f"• {doc}")
        st.caption(f"{docs_data.get('count', 0)} documents indexed")
    except requests.exceptions.ConnectionError:
        st.error("API not reachable. Is uvicorn running?")

    st.divider()
    st.subheader("Upload New Documents")
    uploaded_files = st.file_uploader(
        "Add internal docs (.md, .txt, .html, .pdf)",
        type=["md", "txt", "html", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Uploaded Files"):
        with st.spinner("Uploading and rebuilding indexes..."):
            files_payload = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in uploaded_files
            ]
            try:
                upload_response = requests.post(
                    f"{API_URL}/v1/upload",
                    files=files_payload,
                    timeout=300,
                )
                upload_result = upload_response.json()
                if upload_response.status_code == 200:
                    st.success(upload_result["message"])
                    if upload_result.get("skipped"):
                        st.warning(f"Skipped unsupported files: {', '.join(upload_result['skipped'])}")
                else:
                    st.error(upload_result.get("detail", "Upload failed"))
            except requests.exceptions.ConnectionError:
                st.error("API not reachable. Is uvicorn running?")
            except requests.exceptions.Timeout:
                st.error("Upload timed out — indexing can take a while with many/large files.")

question = st.text_input("Ask a question about your internal docs:", placeholder="e.g. what should I do if a production service goes down")

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and generating answer..."):
        try:
            response = requests.post(
                f"{API_URL}/v1/ask",
                json={"question": question, "strategy": strategy, "top_k": top_k},
                timeout=120,
            )
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Could not reach the API. Make sure uvicorn is running on port 8000.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out. Local inference can be slow on complex queries.")
            st.stop()

    if response.status_code != 200:
        st.error(result.get("detail", "Something went wrong calling the API."))
        st.stop()

    if result.get("status") == "low_confidence":
        st.warning(result["answer"])
        st.subheader("Documents worth checking manually")
        for doc in result.get("documents_worth_checking_manually", []):
            st.text(f"• {doc}")
    else:
        st.subheader("Answer")
        st.write(result["answer"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sources Used")
            sources = result.get("sources", [])
            unique_sources = sorted(set(sources))
            for src in unique_sources:
                st.text(f"• {src}")

        with col2:
            st.subheader("Confidence Breakdown")
            scores = result.get("confidence_scores", {})
            st.metric("Composite Score", scores.get("composite_score", "N/A"))
            st.metric("Retrieval Confidence", scores.get("retrieval_confidence", "N/A"))
            citation_cov = scores.get("citation_coverage", {})
            st.metric("Verified Citation Ratio", citation_cov.get("verified_citation_ratio", "N/A"))
            completeness = scores.get("completeness", {})
            st.metric("Completeness Score", completeness.get("completeness_score", "N/A"))

        with st.expander("Full citation verification report"):
            for entry in scores.get("verification_report", []):
                status_icon = "✅" if entry.get("supported") else "❌"
                st.write(f"{status_icon} **[{entry['citation']}]** {entry['claim']}")
                st.caption(entry.get("reason", ""))