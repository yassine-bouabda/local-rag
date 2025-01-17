# ui/app.py
import streamlit as st
import requests

# Backend URL
API_URL = "http://backend:8000"
SORT_CRITERIA = {
    "Relevance": "relevance",
    "Last Updated Date": "lastUpdatedDate",
    "Submitted Date": "submittedDate",
}
st.title("AI Research Assistant")

# Sidebar: Upload PDF
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload an AI Paper (PDF)", type=["pdf"])

if uploaded_file:
    with st.sidebar:
        st.write("Processing...")
        files = {"file": uploaded_file}
        response = requests.post(f"{API_URL}/upload_pdf", files=files)
        if response.status_code == 200:
            st.success("PDF uploaded and processed successfully!")
        else:
            st.error("Failed to upload PDF.")

# Sidebar: Fetch ArXiv Papers

st.sidebar.header("Fetch AI Papers from ArXiv")
arxiv_query = st.sidebar.text_input("Search ArXiv (e.g., 'GPT models')")
with st.sidebar:
    criteria = st.selectbox("Select sorting criteria:", list(SORT_CRITERIA.keys()))
    max_results = st.slider("Select the maximum number of results to fetch:", min_value=1, max_value=50, value=5)
if st.sidebar.button("Fetch Papers"):
    if arxiv_query:
        response = requests.get(
            f"{API_URL}/fetch_arxiv",
            params={"query": arxiv_query, "criteria": SORT_CRITERIA[criteria], "max_results": max_results},
        )
        if response.status_code == 200:
            papers = response.json().get("papers", [])
            for paper in papers:
                st.sidebar.markdown(f"**[{paper['title']}]({paper['url']})**")
                st.sidebar.write(f"Published: {paper['published']}")
                st.sidebar.write(paper["summary"])
        else:
            st.sidebar.error("Failed to fetch papers from ArXiv.")

# Main: Query Assistant
st.header("Ask the Research Assistant")
query = st.text_input("Enter your question:")
if st.button("Ask"):
    if query:
        response = requests.post(f"{API_URL}/query", data={"question": query})
        if response.status_code == 200:
            answer = response.json().get("response", "No response.")
            st.write("### Assistant's Answer")
            st.write(answer)
        else:
            st.error("Failed to get a response from the assistant.")
