# Principia

Principia is a local knowledge engine for ingesting documents, chunking text, and retrieving relevant information using semantic similarity.

## Features

- Upload PDF and DOCX files
- Split long documents into searchable chunks
- Compute embeddings with SentenceTransformers
- Search using semantic similarity and TF-IDF fallback
- Explore document metadata and tag-based retrieval
- Local-only workflow designed for privacy and offline usage

## Requirements

- Python 3.10+
- pip

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run principia.py
```

## Project structure

- `principia.py` — main Streamlit application
- `requirements.txt` — project dependencies

## Notes

This project stores local brain data in pickle files generated during runtime. These files are ignored by Git by default via `.gitignore`.
