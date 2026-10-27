import os
import json
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_BRAIN = {
    "texts": [],
    "embeddings": [],
    "sources": [],
    "timestamps": [],
    "tags": [],
    "metadata": {},
}


# ===========================================================
# App configuration
# ===========================================================
st.set_page_config(
    page_title="Principia",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0b1022 0%, #161d33 100%);
        color: white;
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(90deg, #4f46e5 0%, #8b5cf6 100%);
        padding: 1.6rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.3);
    }

    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }

    .chat-message {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #8b5cf6;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0;
    }

    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.35);
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background: rgba(255,255,255,0.05);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .stTextInput label,
    .stTextArea label,
    .stSelectbox label {
        color: white !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: rgba(255,255,255,0.6) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================
# Data normalization and timestamp safety
# ===========================================================
def normalize_brain_data(data):
    """
    Repair a saved brain dictionary so text, sources, timestamps, tags, and embeddings
    stay aligned and valid, even if older pickles are partially broken.
    """
    if not isinstance(data, dict):
        return DEFAULT_BRAIN.copy()

    normalized = DEFAULT_BRAIN.copy()
    normalized.update(data)

    texts = list(normalized.get("texts") or [])
    sources = list(normalized.get("sources") or [])
    timestamps = list(normalized.get("timestamps") or [])
    tags = list(normalized.get("tags") or [])
    embeddings = list(normalized.get("embeddings") or [])

    # Coerce common weird cases
    texts = [str(x) if x is not None else "" for x in texts]
    sources = [str(x) if x is not None else "unknown" for x in sources]
    timestamps = [x for x in timestamps]
    tags = [list(x) if isinstance(x, (list, tuple, set)) else [str(x)] if x not in (None, "") else [] for x in tags]

    target_len = max(len(texts), len(sources), len(timestamps), len(tags))

    if len(texts) < target_len:
        texts.extend([""] * (target_len - len(texts)))
    if len(sources) < target_len:
        sources.extend(["unknown"] * (target_len - len(sources)))
    if len(timestamps) < target_len:
        timestamps.extend([None] * (target_len - len(timestamps)))
    if len(tags) < target_len:
        tags.extend([[] for _ in range(target_len - len(tags))])

    if len(embeddings) > len(texts):
        embeddings = embeddings[: len(texts)]
    if len(embeddings) < len(texts):
        embeddings.extend([np.array([], dtype=float)] * (len(texts) - len(embeddings)))

    normalized["texts"] = texts[:target_len]
    normalized["sources"] = sources[:target_len]
    normalized["timestamps"] = timestamps[:target_len]
    normalized["tags"] = tags[:target_len]
    normalized["embeddings"] = embeddings[:target_len]
    return normalized


def coerce_datetime_series(values):
    """
    Convert timestamp-like values to a pandas datetime series while tolerating:
    - empty data
    - None/NaT
    - mixed string formats
    - invalid values
    - dates in different formats
    """
    if values is None:
        return pd.Series([], dtype="datetime64[ns]")

    if isinstance(values, pd.Series):
        series = values
    elif isinstance(values, pd.Index):
        series = pd.Series(values)
    elif isinstance(values, (list, tuple, np.ndarray)):
        series = pd.Series(values)
    else:
        series = pd.Series([values])

    parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed, pd.DatetimeIndex):
        parsed = pd.Series(parsed)

    return parsed.dropna().reset_index(drop=True)


def build_ingestion_timeline(raw_timestamps):
    """
    Build a date-based ingestion timeline without crashing on invalid or mixed data.
    This is the exact fix for the `.dt.date` AttributeError.
    """
    timestamps = coerce_datetime_series(raw_timestamps)
    if timestamps.empty:
        return pd.DataFrame(
            {"date": pd.Series([], dtype="datetime64[ns]"), "count": pd.Series([], dtype="int64")}
        )

    timeline = pd.DataFrame({"date": timestamps.dt.date, "count": 1}).dropna(subset=["date"])
    if timeline.empty:
        return pd.DataFrame(
            {"date": pd.Series([], dtype="datetime64[ns]"), "count": pd.Series([], dtype="int64")}
        )

    timeline = timeline.groupby("date", as_index=False).size().rename(columns={"size": "count"})
    timeline["date"] = pd.to_datetime(timeline["date"])
    return timeline.sort_values("date").reset_index(drop=True)


def format_timestamp_for_display(value):
    """
    Format timestamps for UI display, while safely handling None / invalid string / mixed data.
    """
    if value is None or value == "" or value == "NaT":
        return "N/A"

    try:
        if isinstance(value, pd.Timestamp):
            return value.isoformat(timespec="seconds")
        if isinstance(value, datetime):
            return value.isoformat(timespec="seconds")
        if isinstance(value, str):
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return "N/A"
            return pd.Timestamp(parsed).isoformat(timespec="seconds")
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return "N/A"
        return pd.Timestamp(parsed).isoformat(timespec="seconds")
    except Exception:
        return str(value)[:19] if isinstance(value, str) else "N/A"


def normalize_text_for_matching(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text):
    if not text or not text.strip():
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def summarize_text(text, max_sentences=3):
    if not text or not text.strip():
        return "No content available to summarize."

    sentences = split_into_sentences(text)
    if not sentences:
        return text.strip()[:500]

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    keyword_tokens = set(normalize_text_for_matching(text).split())
    scored = []

    for sentence in sentences:
        s_norm = normalize_text_for_matching(sentence)
        if not s_norm:
            continue
        overlap = len(set(s_norm.split()) & keyword_tokens)
        score = overlap + len(sentence.split()) * 0.02
        scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [sentence for _, sentence in scored[:max_sentences]]
    return " ".join(selected)


def build_answer_from_results(question, results):
    if not results:
        return "I couldn't find relevant information in your documents."

    question_lower = question.lower()

    if any(word in question_lower for word in [
        "summarize",
        "summary",
        "overview",
        "what is this document about",
        "tell me about",
        "briefly explain",
    ]):
        combined = "\n\n".join(r["text"] for r in results[:5])
        return summarize_text(combined, max_sentences=3)

    context = "\n\n".join(r["text"] for r in results[:3])
    q_tokens = set(normalize_text_for_matching(question).split())
    best_sentence = ""
    best_score = -1

    for sentence in split_into_sentences(context):
        s_tokens = set(normalize_text_for_matching(sentence).split())
        score = len(q_tokens & s_tokens)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if not best_sentence:
        return summarize_text(context, max_sentences=2)

    return best_sentence


# ===========================================================
# Brain storage
# ===========================================================
class KnowledgeBrain:
    def __init__(self, brain_name="default"):
        self.brain_name = brain_name
        self.db_file = f"principia_brain_{brain_name}.pkl"
        self.data = self.load()

    def load(self):
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, "rb") as f:
                    loaded = pickle.load(f)
                return normalize_brain_data(loaded)
        except Exception:
            pass
        return DEFAULT_BRAIN.copy()

    def save(self):
        with open(self.db_file, "wb") as f:
            pickle.dump(self.data, f)

    def add_chunk(self, text, source, tags=None):
        self.data["texts"].append(text)
        self.data["sources"].append(source)
        self.data["timestamps"].append(datetime.now().isoformat())
        self.data["tags"].append(tags or [])
        self.save()

    def add_embeddings(self, embeddings):
        if isinstance(embeddings, np.ndarray):
            self.data["embeddings"].extend(embeddings.tolist())
        else:
            self.data["embeddings"].extend(list(embeddings))
        self.save()

    def get_stats(self):
        valid_timestamps = [ts for ts in self.data.get("timestamps", []) if ts not in (None, "", "NaT")]
        created = "N/A"

        if valid_timestamps:
            try:
                parsed = pd.to_datetime(valid_timestamps, errors="coerce").dropna()
                if not parsed.empty:
                    created = pd.Timestamp(parsed.min()).isoformat()
            except Exception:
                created = "N/A"

        return {
            "total_chunks": len(self.data.get("texts", [])),
            "total_docs": len(set(self.data.get("sources", []))),
            "total_tags": len(set(tag for tag_list in self.data.get("tags", []) for tag in tag_list)),
            "created": created,
        }


def ensure_embeddings(brain, model):
    """
    Rebuild embeddings when the saved brain has mismatched or incomplete embedding vectors.
    """
    if model is None:
        return False

    texts = brain.data.get("texts") or []
    if not texts:
        return True

    embeddings = brain.data.get("embeddings") or []
    valid_count = len(embeddings)

    if len(embeddings) == len(texts):
        valid = True
        for emb in embeddings:
            try:
                arr = np.asarray(emb, dtype=float)
                if arr.size == 0:
                    valid = False
                    break
            except Exception:
                valid = False
                break
        if valid:
            return True

    rebuilt = []
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            rebuilt.append(np.zeros(384, dtype=float))
            continue

        existing = embeddings[i] if i < valid_count else None
        try:
            arr = np.asarray(existing, dtype=float)
            if arr.size > 0:
                rebuilt.append(arr)
            else:
                rebuilt.append(np.asarray(model.encode([text])[0], dtype=float))
        except Exception:
            rebuilt.append(np.asarray(model.encode([text])[0], dtype=float))

    brain.data["embeddings"] = rebuilt
    brain.save()
    return True


# ===========================================================
# Model loading
# ===========================================================
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Unable to load sentence-transformer model: {e}")
        return None


# ===========================================================
# Document ingestion
# ===========================================================
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""


def smart_chunk_text(text, chunk_size=500):
    if not text or not text.strip():
        return []

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_words = para.split()
        para_length = len(para_words)

        if current_length + para_length > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# ===========================================================
# Search and retrieval
# ===========================================================
def semantic_search(query, brain, model, top_k=5):
    if not query or len(brain.data["texts"]) == 0:
        return [], []

    ensure_embeddings(brain, model)
    if not brain.data.get("embeddings") or len(brain.data["embeddings"]) != len(brain.data["texts"]):
        return [], []

    try:
        query_embedding = model.encode([query]) if model is not None else None
        if query_embedding is None:
            return [], []
        similarities = cosine_similarity(query_embedding, np.asarray(brain.data["embeddings"]))[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
    except Exception:
        return [], []

    results = []
    for idx in top_indices:
        if idx >= len(brain.data["texts"]):
            continue
        results.append({
            "text": brain.data["texts"][idx],
            "source": brain.data["sources"][idx],
            "timestamp": brain.data["timestamps"][idx],
            "tags": brain.data["tags"][idx],
            "score": float(similarities[idx]),
        })
    return results, [float(similarities[i]) for i in top_indices if i < len(brain.data["texts"])]


def keyword_search(query, brain, top_k=5):
    if not query or len(brain.data["texts"]) == 0:
        return [], []

    valid_texts = [text for text in brain.data["texts"] if isinstance(text, str)]
    if not valid_texts:
        return [], []

    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(valid_texts)
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
    except Exception:
        return [], []

    results = []
    scores = []

    for idx in top_indices:
        if idx >= len(valid_texts):
            continue

        actual_index = None
        for i, text in enumerate(brain.data["texts"]):
            if isinstance(text, str) and text == valid_texts[idx]:
                actual_index = i
                break

        if actual_index is None:
            continue

        results.append({
            "text": brain.data["texts"][actual_index],
            "source": brain.data["sources"][actual_index],
            "timestamp": brain.data["timestamps"][actual_index],
            "tags": brain.data["tags"][actual_index],
            "score": float(similarities[idx]),
        })
        scores.append(float(similarities[idx]))

    return results, scores


def hybrid_search(query, brain, model, top_k=5, semantic_weight=0.7):
    if not query:
        return [], []

    semantic_results, semantic_scores = semantic_search(query, brain, model, top_k * 2)
    keyword_results, keyword_scores = keyword_search(query, brain, top_k * 2)

    combined = {}

    for i, result in enumerate(semantic_results):
        key = result["text"][:120]
        combined[key] = {
            "text": result["text"],
            "source": result["source"],
            "timestamp": result["timestamp"],
            "tags": result["tags"],
            "score": semantic_scores[i] * semantic_weight,
        }

    for i, result in enumerate(keyword_results):
        key = result["text"][:120]
        if key in combined:
            combined[key]["score"] += keyword_scores[i] * (1 - semantic_weight)
        else:
            combined[key] = {
                "text": result["text"],
                "source": result["source"],
                "timestamp": result["timestamp"],
                "tags": result["tags"],
                "score": keyword_scores[i] * (1 - semantic_weight),
            }

    if not combined:
        return [], []

    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    scores = [item["score"] for item in sorted_results]
    return sorted_results, scores


# ===========================================================
# Main app
# ===========================================================
def main():
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin: 0; color: white;">🧠 Principia</h1>
            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
                Local knowledge engine for PDFs, Word docs, notes, and research material
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "brain_options" not in st.session_state:
        st.session_state.brain_options = ["default", "work", "research", "personal"]

    with st.sidebar:
        st.header("🧠 Brain Manager")

        selected_brain = st.selectbox("Choose Brain", st.session_state.brain_options)
        brain = KnowledgeBrain(selected_brain)

        new_brain_name = st.text_input("New brain name", placeholder="e.g. legal")
        if st.button("Create Brain") and new_brain_name.strip():
            clean_name = new_brain_name.strip()
            if clean_name not in st.session_state.brain_options:
                st.session_state.brain_options.append(clean_name)
                selected_brain = clean_name
                st.rerun()

        st.divider()
        st.header("📥 Ingest Documents")

        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
        )

        tags_input = st.text_input("Tags (comma-separated)", placeholder="e.g. research, q4, legal")
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()] if tags_input else []

        if st.button("Process & Store"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                model = load_embedding_model()
                with st.spinner("Processing documents..."):
                    progress = st.progress(0)
                    total_files = len(uploaded_files)

                    for i, file in enumerate(uploaded_files):
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.getvalue())

                        if file.name.lower().endswith(".pdf"):
                            text = extract_text_from_pdf(temp_path)
                        elif file.name.lower().endswith(".docx"):
                            text = extract_text_from_docx(temp_path)
                        else:
                            try:
                                raw = file.getvalue()
                                text = raw.decode("utf-8")
                            except Exception:
                                text = file.getvalue().decode("utf-8", errors="ignore")

                        os.remove(temp_path)

                        chunks = smart_chunk_text(text)
                        if not chunks:
                            chunks = [text.strip()] if text.strip() else [""]

                        all_embeddings = []

                        for chunk in chunks:
                            if not chunk.strip():
                                continue
                            brain.add_chunk(chunk, file.name, tags)
                            if model is not None:
                                emb = model.encode([chunk])[0]
                                all_embeddings.append(emb)

                        if model is not None and all_embeddings:
                            brain.add_embeddings(np.array(all_embeddings))

                        progress.progress((i + 1) / total_files)

                    brain.data = normalize_brain_data(brain.data)
                    st.success(f"Successfully processed {len(uploaded_files)} file(s).")
                    st.balloons()
                    st.rerun()

        st.divider()
        st.header("📊 Stats")

        stats = brain.get_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", stats["total_chunks"])
            st.metric("Docs", stats["total_docs"])
        with col2:
            st.metric("Tags", stats["total_tags"])
            st.metric("Created", stats["created"][:10] if stats["created"] != "N/A" else "N/A")

        st.divider()
        st.caption("🔒 Local-only. No external API calls.")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "💬 Chat", "📊 Analytics", "⚙️ Settings"])

    # Search tab
    with tab1:
        st.header("🔍 Search your knowledge")
        query = st.text_input("Ask a question", placeholder="What is the key finding in the report?")

        search_mode = st.selectbox("Mode", ["Hybrid (Best)", "Semantic", "Keyword"])
        top_k = st.slider("Results", min_value=1, max_value=10, value=5)

        if st.button("Search") and query.strip():
            if len(brain.data["texts"]) == 0:
                st.warning("Your brain is empty. Upload a document first.")
            else:
                brain.data = normalize_brain_data(brain.data)
                model = load_embedding_model()

                with st.spinner("Searching..."):
                    if search_mode == "Semantic":
                        results, scores = semantic_search(query, brain, model, top_k)
                    elif search_mode == "Keyword":
                        results, scores = keyword_search(query, brain, top_k)
                    else:
                        results, scores = hybrid_search(query, brain, model, top_k)

                if results:
                    summary = summarize_text("\n\n".join(r["text"] for r in results[:5]), max_sentences=3)
                    st.markdown(f"**Summary:** {summary}")

                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} — {result['source']}"):
                            st.markdown(f"**Source:** `{result['source']}`")
                            st.markdown(f"**Time:** {format_timestamp_for_display(result['timestamp'])}")
                            if result["tags"]:
                                st.markdown("**Tags:** " + " ".join([f"`{tag}`" for tag in result["tags"]]))
                            st.markdown("---")
                            st.write(result["text"])
                else:
                    st.info("No relevant results found. Try a broader question or add more documents.")

    # Chat tab
    with tab2:
        st.header("💬 Ask the knowledge base")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            role = message["role"]
            with st.container():
                st.markdown(
                    f"""
                    <div class="chat-message">
                        <strong>{'🧑 You' if role == 'user' else '🤖 Brain'}:</strong><br>
                        {message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        user_query = st.text_input("Type a question", placeholder="Summarize the main points...")

        if st.button("Send") and user_query.strip():
            if len(brain.data["texts"]) == 0:
                st.warning("Your brain is empty. Upload a document first.")
            else:
                brain.data = normalize_brain_data(brain.data)
                model = load_embedding_model()

                st.session_state.chat_history.append({"role": "user", "content": user_query.strip()})

                with st.spinner("Thinking..."):
                    results, scores = hybrid_search(user_query.strip(), brain, model, top_k=5)
                    if results:
                        answer = build_answer_from_results(user_query.strip(), results)
                    else:
                        answer = "I couldn't find relevant information in your documents. Try rephrasing the question or upload more documents."

                st.session_state.chat_history.append({"role": "brain", "content": answer})
                st.rerun()

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Analytics tab
    with tab3:
        st.header("📊 Analytics")

        if len(brain.data["texts"]) == 0:
            st.info("Upload documents to see analytics.")
        else:
            st.subheader("Document sources")
            source_counts = pd.Series(brain.data["sources"]).value_counts()
            fig_sources = px.pie(values=source_counts.values, names=source_counts.index, title="Documents by Source")
            st.plotly_chart(fig_sources, use_container_width=True)

            st.subheader("Ingestion timeline")
            timeline = build_ingestion_timeline(brain.data["timestamps"])
            if timeline.empty:
                st.info("No valid timestamps available yet.")
            else:
                fig_timeline = px.line(
                    timeline,
                    x="date",
                    y="count",
                    title="Documents added over time",
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

            if any(brain.data["tags"]):
                st.subheader("Tag distribution")
                all_tags = [tag for tag_list in brain.data["tags"] for tag in tag_list]
                if all_tags:
                    tag_counts = pd.Series(all_tags).value_counts().head(20)
                    fig_tags = px.bar(
                        x=tag_counts.values,
                        y=tag_counts.index,
                        orientation="h",
                        title="Top tags",
                    )
                    st.plotly_chart(fig_tags, use_container_width=True)

    # Settings tab
    with tab4:
        st.header("⚙️ Settings")

        st.subheader("Export brain data")
        if st.button("Export JSON"):
            export_data = {
                "brain_name": selected_brain,
                "stats": brain.get_stats(),
                "documents": [
                    {
                        "text": text,
                        "source": source,
                        "timestamp": timestamp,
                        "tags": tags,
                    }
                    for text, source, timestamp, tags in zip(
                        brain.data["texts"],
                        brain.data["sources"],
                        brain.data["timestamps"],
                        brain.data["tags"],
                    )
                ],
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"principia_{selected_brain}_export.json",
                mime="application/json",
            )

        st.divider()

        st.subheader("Danger zone")
        if st.button("Clear this brain", type="secondary"):
            if st.checkbox("I understand this will delete the current brain data."):
                brain.data = DEFAULT_BRAIN.copy()
                brain.save()
                st.success("Brain cleared.")
                st.rerun()

        st.divider()
        st.info(
            """
            **Notes**
            - This app stores data locally in a pickle file.
            - Search and chat are based on the uploaded content in the local brain.
            - No external API or cloud processing is required.
            """
        )


if __name__ == "__main__":
    main()
