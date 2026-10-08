import os
import time
import hashlib

import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="Smart Document Assistant",
    page_icon="📚",
    layout="wide"
)


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY was not found in the .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

def extract_pdf(file):
    documents = []

    reader = PdfReader(file)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append({
                "text": text.strip(),
                "source": file.name,
                "page": page_number
            })

    return documents


def extract_txt(file):
    text = file.read().decode("utf-8", errors="ignore")

    if text.strip():
        return [{
            "text": text.strip(),
            "source": file.name,
            "page": None
        }]

    return []


def create_chunks(documents, chunk_size=800, overlap=120):

    chunks = []

    for document in documents:

        text = document["text"]
        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": document["source"],
                    "page": document["page"]
                })

            start += chunk_size - overlap

    return chunks


def process_files(uploaded_files):

    documents = []

    for file in uploaded_files:

        if file.name.lower().endswith(".pdf"):
            documents.extend(extract_pdf(file))

        elif file.name.lower().endswith(".txt"):
            documents.extend(extract_txt(file))

    chunks = create_chunks(documents)

    return chunks


def make_file_signature(uploaded_files):

    data = ""

    for file in uploaded_files:
        file_bytes = file.getvalue()

        data += file.name
        data += str(len(file_bytes))
        data += hashlib.md5(file_bytes).hexdigest()

    return hashlib.md5(data.encode()).hexdigest()

def build_index(chunks):
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(texts)

    return vectorizer, vectors

def search_documents(question, index, chunks, top_k=5):

    vectorizer, vectors = index

    question_vector = vectorizer.transform([question])

    scores = (vectors @ question_vector.T).toarray().flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index_number in top_indices:

        results.append({
            "chunk": chunks[index_number],
            "distance": float(scores[index_number])
        })

    return results

def generate_answer(question, results, messages):

    context_parts = []

    for result in results:

        chunk = result["chunk"]

        page_text = ""

        if chunk["page"]:
            page_text = f"Page {chunk['page']}"

        context_pa