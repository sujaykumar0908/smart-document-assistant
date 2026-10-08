# 📚 Smart Document Knowledge Assistant

A Python-based document question-answering system that allows users to upload PDF and TXT lecture notes and ask questions about them.

## Features

- Upload multiple PDF and TXT documents
- Extract text from documents
- Split documents into searchable chunks
- Generate semantic embeddings
- Perform similarity search using FAISS
- Retrieve the top 5 relevant chunks
- Generate answers using Google Gemini
- Display referenced document snippets
- Maintain chat history
- Support follow-up questions
- Streamlit web interface

## Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Google Gemini API

## How It Works

1. User uploads lecture documents.
2. Text is extracted from the documents.
3. Documents are divided into smaller chunks.
4. Sentence Transformers converts the chunks into embeddings.
5. FAISS performs similarity search.
6. The most relevant chunks are retrieved.
7. The retrieved context is sent to Gemini.
8. Gemini generates an answer using the document context.
9. Relevant source snippets are displayed.

## Installation

Create a virtual environment:

```bash
python -m venv venv