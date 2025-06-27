import streamlit as st
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import openai

# Load .env variables
load_dotenv()
AZURE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set API keys
openai.api_key = OPENAI_API_KEY
client = TextAnalyticsClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))

# Streamlit config
st.set_page_config(page_title="🧠 Smart Summarizer & Q&A", layout="centered")
st.title("🧠 Smart Summarizer & Q&A App")

tab1, tab2 = st.tabs(["📝 Summarization", "💬 Q&A"])

# ---------- Extractors ----------
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        return "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)
    except Exception as e:
        return f"Error extracting text: {e}"

def extract_text_from_pdf(uploaded_file):
    try:
        text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# ---------- Azure Summarizer ----------
def summarize_with_azure(text):
    try:
        chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
        full_summary = []
        for chunk in chunks:
            poller = client.begin_extract_summary([chunk], language="en")
            result = list(poller.result())
            if result and hasattr(result[0], "sentences"):
                full_summary.extend([s.text for s in result[0].sentences])
        return full_summary
    except Exception as e:
        return [f"Summarization failed: {e}"]

# ---------- OpenAI Q&A ----------
def ask_question_openai(question, context):
    try:
        prompt = f"""You are a helpful assistant. Answer the following question based on the given context.

Context:
{context}

Question: {question}
Answer:"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ OpenAI Error: {e}"

# ---------- TAB 1: Summarization ----------
with tab1:
    st.subheader("Summarize from URL or PDF")
    input_mode1 = st.radio("Choose input type for summarization:", ["🔗 URL", "📄 PDF"])
    text1 = ""

    if input_mode1 == "🔗 URL":
        url1 = st.text_input("Enter article URL for summary:")
        if st.button("Summarize URL"):
            with st.spinner("Extracting..."):
                text1 = extract_text_from_url(url1)
            if text1 and len(text1) > 100:
                with st.spinner("Summarizing..."):
                    summary = summarize_with_azure(text1)
                st.success("Summary:")
                for sentence in summary:
                    st.write("→", sentence)
            else:
                st.warning("Couldn't extract enough content.")

    elif input_mode1 == "📄 PDF":
        uploaded_file1 = st.file_uploader("Upload a PDF for summary", type=["pdf"], key="summary_pdf")
        if uploaded_file1 and st.button("Summarize PDF"):
            with st.spinner("Extracting text..."):
                text1 = extract_text_from_pdf(uploaded_file1)
            if text1 and len(text1) > 100:
                with st.spinner("Summarizing..."):
                    summary = summarize_with_azure(text1)
                st.success("Summary:")
                for sentence in summary:
                    st.write("→", sentence)
            else:
                st.warning("Couldn't extract enough content.")

# ---------- TAB 2: Q&A ----------
with tab2:
    st.subheader("Ask Questions Based on URL or PDF")
    input_mode2 = st.radio("Choose input type for Q&A:", ["🔗 URL", "📄 PDF"], key="qa_radio")
    text2 = ""

    if input_mode2 == "🔗 URL":
        url2 = st.text_input("Enter article URL for Q&A:")
        if url2:
            with st.spinner("Extracting..."):
                text2 = extract_text_from_url(url2)

    elif input_mode2 == "📄 PDF":
        uploaded_file2 = st.file_uploader("Upload a PDF for Q&A", type=["pdf"], key="qa_pdf")
        if uploaded_file2:
            with st.spinner("Extracting text..."):
                text2 = extract_text_from_pdf(uploaded_file2)

    if text2 and len(text2) > 100:
        question = st.text_input("Your question:")
        if question:
            with st.spinner("Getting answer from OpenAI..."):
                answer = ask_question_openai(question, text2)
                st.success(f"Answer: {answer}")
    elif text2:
        st.warning("Extracted content is too short.")
