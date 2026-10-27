import os
import streamlit as st
import numpy as np
import pandas as pd
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from datetime import datetime
import plotly.express as px
import re


DEFAULT_BRAIN = {
    "texts": [],
    "embeddings": [],
    "sources": [],
    "timestamps": [],
    "tags": [],
    "metadata": {},
}


# ==========================================
# 1. CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="Principia 2.0 | Ultimate Local Knowledge Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for stunning UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #e8eaf6;
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
    }

    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #667eea;
    }

    .source-tag {
        display: inline-block;
        background: rgba(102, 126, 234, 0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin: 0.2rem;
    }

    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
    }

    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e8eaf6;
        border-radius: 10px;
    }

    .stExpander {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_brain_data(data):
    """Repair a brain dict so all required lists stay aligned and valid."""
    if not isinstance(data, dict):
        return DEFAULT_BRAIN.copy()

    normalized = DEFAULT_BRAIN.copy()
    normalized.update(data)

    texts = list(normalized.get("texts") or [])
    sources = list(normalized.get("sources") or [])
    timestamps = list(normalized.get("timestamps") or [])
    tags = list(normalized.get("tags") or [])
    embeddings = list(normalized.get("embeddings") or [])

    target_len = max(len(texts), len(sources), len(timestamps), len(tags))

    if len(sources) < target_len:
        sources.extend(["unknown"] * (target_len - len(sources)))
    if len(timestamps) < target_len:
        timestamps.extend([None] * (target_len - len(timestamps)))
    if len(tags) < target_len:
        tags.extend([[] for _ in range(target_len - len(tags))])

    if len(texts) < target_len:
        texts.extend([""] * (target_len - len(texts)))

    if len(embeddings) > len(texts):
        embeddings = embeddings[: len(texts)]
    if len(embeddings) < len(texts):
        embeddings.extend([np.array([])] * (len(texts) - len(embeddings)))

    normalized["texts"] = texts
    normalized["sources"] = sources[: len(texts)]
    normalized["timestamps"] = timestamps[: len(texts)]
    normalized["tags"] = tags[: len(texts)]
    normalized["embeddings"] = embeddings[: len(texts)]

    return normalized


def coerce_datetime_series(values):
    """Normalize timestamp-like values from persisted brain data into a pandas Series."""
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
    """Build a date-based ingestion timeline while tolerating invalid or missing data."""
    timestamps = coerce_datetime_series(raw_timestamps)
    if timestamps.empty:
        return pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"), "count": pd.Series([], dtype="int64")})

    timeline = pd.DataFrame({"date": timestamps.dt.date, "count": 1}).dropna(subset=["date"])
    if timeline.empty:
        return pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"), "count": pd.Series([], dtype="int64")})

    timeline = timeline.groupby("date", as_index=False).size().rename(columns={"size": "count"})
    timeline["date"] = pd.to_datetime(timeline["date"])
    return timeline.sort_values("date").reset_index(drop=True)


def format_timestamp_for_display(value):
    """Safely convert timestamps to display strings without crashing on bad data."""
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


# ==========================================
# 2. DATABASE MANAGEMENT
# ==========================================
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
        self.data["embeddings"].extend(embeddings.tolist())
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
            "total_tags": len(set(tag for tags in self.data.get("tags", []) for tag in tags)),
            "created": created,
        }


# ==========================================
# 3. AI MODEL LOADING
# ==========================================
@st.cache_resource
def load_embedding_model():
    """Load the local embedding model (downloads once, then 100% offline)"""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ==========================================
# 4. DOCUMENT PROCESSING
# ==========================================
def extract_text_from_pdf(file_path):
    """Extract text from PDF files"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            text += (extracted or "") + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def extract_text_from_docx(file_path):
    """Extract text from Word documents"""
    try:
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for better context"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def smart_chunk_text(text, chunk_size=500):
    """Intelligent chunking that respects paragraph boundaries"""
    if not text or not text.strip():
        return []

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_length = len(para.split())
        if current_length + para_length > chunk_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# ==========================================
# 5. SEARCH & RETRIEVAL
# ==========================================
def semantic_search(query, brain, model, top_k=5):
    """Perform semantic search using embeddings"""
    if not query or len(brain.data["texts"]) == 0:
        return [], []

    if not brain.data.get("embeddings") or len(brain.data["embeddings"]) != len(brain.data["texts"]):
        return [], []

    try:
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, np.array(brain.data["embeddings"]))[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
    except Exception:
        return [], []

    results = []
    scores = []
    for idx in top_indices:
        results.append(
            {
                "text": brain.data["texts"][idx],
                "source": brain.data["sources"][idx],
                "timestamp": brain.data["timestamps"][idx],
                "tags": brain.data["tags"][idx],
                "score": float(similarities[idx]),
            }
        )
        scores.append(float(similarities[idx]))

    return results, scores


def keyword_search(query, brain, top_k=5):
    """Fallback keyword-based search using TF-IDF"""
    if not query or len(brain.data["texts"]) == 0:
        return [], []

    try:
        texts = [text for text in brain.data["texts"] if isinstance(text, str)]
        if not texts:
            return [], []

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
    except Exception:
        return [], []

    results = []
    scores = []
    for idx in top_indices:
        actual_index = idx
        results.append(
            {
                "text": brain.data["texts"][actual_index],
                "source": brain.data["sources"][actual_index],
                "timestamp": brain.data["timestamps"][actual_index],
                "tags": brain.data["tags"][actual_index],
                "score": float(similarities[actual_index]),
            }
        )
        scores.append(float(similarities[actual_index]))

    return results, scores


def hybrid_search(query, brain, model, top_k=5, semantic_weight=0.7):
    """Combine semantic and keyword search for best results"""
    if not query:
        return [], []

    semantic_results, semantic_scores = semantic_search(query, brain, model, top_k * 2)
    keyword_results, keyword_scores = keyword_search(query, brain, top_k * 2)

    combined = {}

    for i, result in enumerate(semantic_results):
        key = result["text"][:100]
        combined[key] = {
            "text": result["text"],
            "source": result["source"],
            "timestamp": result["timestamp"],
            "tags": result["tags"],
            "score": semantic_scores[i] * semantic_weight,
        }

    for i, result in enumerate(keyword_results):
        key = result["text"][:100]
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
    scores = [r["score"] for r in sorted_results]
    return sorted_results, scores


# ==========================================
# 6. MAIN APPLICATION
# ==========================================
def main():
    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin: 0; color: white;">🧠 Principia 2.0</h1>
            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
                Ultimate Local Knowledge Engine | 100% Offline | Zero API Calls | Your Data, Your Control
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar - Brain Management
    with st.sidebar:
        st.header("🧠 Knowledge Brains")

        brain_options = ["default", "work", "research", "personal", "custom"]
        selected_brain = st.selectbox("Select Brain", brain_options)

        if st.button("Create New Brain"):
            new_brain_name = st.text_input("Brain name:")
            if new_brain_name:
                brain_options.append(new_brain_name)
                selected_brain = new_brain_name

        brain = KnowledgeBrain(selected_brain)
        brain.data = normalize_brain_data(brain.data)

        st.divider()
        st.header("📥 Ingest Documents")

        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            help="Supports PDF, TXT, Markdown, and Word documents",
        )

        tags_input = st.text_input("Tags (comma-separated)", placeholder="e.g., important, project-x, 2024")
        tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []

        if st.button("Process & Store"):
            if uploaded_files:
                model = load_embedding_model()
                if model is None:
                    st.error("Failed to load AI model")
                    return

                with st.spinner("Processing documents..."):
                    total_chunks = 0
                    progress_bar = st.progress(0)

                    for i, file in enumerate(uploaded_files):
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.getvalue())

                        if file.name.endswith(".pdf"):
                            text = extract_text_from_pdf(temp_path)
                        elif file.name.endswith(".docx"):
                            text = extract_text_from_docx(temp_path)
                        else:
                            try:
                                text = file.getvalue().decode("utf-8")
                            except Exception:
                                text = file.getvalue().decode("utf-8", errors="ignore")

                        os.remove(temp_path)

                        chunks = smart_chunk_text(text)
                        all_embeddings = []

                        for chunk in chunks:
                            brain.add_chunk(chunk, file.name, tags)
                            embedding = model.encode([chunk])
                            all_embeddings.append(embedding[0])

                        if all_embeddings:
                            brain.add_embeddings(np.array(all_embeddings))

                        total_chunks += len(chunks)
                        progress_bar.progress((i + 1) / len(uploaded_files))

                    brain.data = normalize_brain_data(brain.data)
                    st.success(f"✅ Successfully processed {len(uploaded_files)} files ({total_chunks} chunks)")
                    st.balloons()
            else:
                st.warning("Please upload at least one file")

        st.divider()

        st.header("📊 Brain Statistics")
        stats = brain.get_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats["total_chunks"])
            st.metric("Total Tags", stats["total_tags"])
        with col2:
            st.metric("Documents", stats["total_docs"])
            st.metric("Created", stats["created"][:10] if stats["created"] != "N/A" else "N/A")

        st.divider()
        st.caption("🔒 100% Local | No Cloud | No API | Your Privacy Guaranteed")

    # Main Content Area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "💬 Chat", "📈 Analytics", "⚙️ Settings"])

    with tab1:
        st.header("🔍 Intelligent Search")
        st.write("Search across all your documents using natural language")

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "What are you looking for?",
                placeholder="e.g., What did the Q3 report say about revenue growth?",
                label_visibility="collapsed",
            )
        with col2:
            search_mode = st.selectbox("Search Mode", ["Hybrid (Best)", "Semantic", "Keyword"])
            top_k = st.slider("Results", 1, 10, 5)

        if st.button("Search") and query:
            if len(brain.data["texts"]) == 0:
                st.warning("Your brain is empty! Upload documents first.")
            else:
                brain.data = normalize_brain_data(brain.data)
                model = load_embedding_model()
                if model is None:
                    st.error("Failed to load AI model")
                    return

                with st.spinner("Searching..."):
                    if search_mode == "Semantic":
                        results, scores = semantic_search(query, brain, model, top_k)
                    elif search_mode == "Keyword":
                        results, scores = keyword_search(query, brain, top_k)
                    else:
                        results, scores = hybrid_search(query, brain, model, top_k)

                if results:
                    st.success(f"Found {len(results)} relevant results")

                    for i, result in enumerate(results, 1):
                        with st.expander(f"📄 Result {i} (Score: {result['score']:.3f}) - {result['source']}"):
                            st.markdown(f"**Source:** {result['source']}")
                            st.markdown(f"**Time:** {format_timestamp_for_display(result['timestamp'])}")

                            if result["tags"]:
                                st.markdown("**Tags:** " + " ".join([f"`{tag}`" for tag in result["tags"]]))

                            st.markdown("---")
                            st.write(result["text"])
                else:
                    st.info("No results found. Try different keywords or upload more documents.")

    with tab2:
        st.header("💬 Knowledge Chat")
        st.write("Ask questions and get answers from your documents")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.container():
                st.markdown(
                    f"""
                    <div class="chat-message">
                        <strong>{'🧑 You' if message['role'] == 'user' else '🤖 Brain'}:</strong><br>
                        {message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        user_query = st.text_input(
            "Ask a question:",
            placeholder="e.g., Summarize the key findings from the research papers",
            label_visibility="collapsed",
        )

        if st.button("Send") and user_query:
            if len(brain.data["texts"]) == 0:
                st.warning("Your brain is empty! Upload documents first.")
            else:
                brain.data = normalize_brain_data(brain.data)
                model = load_embedding_model()
                if model is None:
                    st.error("Failed to load AI model")
                    return

                st.session_state.chat_history.append({"role": "user", "content": user_query})

                with st.spinner("Thinking..."):
                    results, scores = hybrid_search(user_query, brain, model, top_k=5)

                    if results:
                        context = "\n\n".join([f"[{r['source']}] {r['text']}" for r in results[:3]])
                        response = (
                            f"Based on your documents, here's what I found:\n\n{context}\n\n"
                            f"**Relevance Scores:** {', '.join([f'{s:.3f}' for s in scores[:3]])}"
                        )
                    else:
                        response = "I couldn't find relevant information in your documents. Try rephrasing your question or uploading more documents."

                st.session_state.chat_history.append({"role": "brain", "content": response})
                st.rerun()

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    with tab3:
        st.header("📈 Analytics & Insights")

        if len(brain.data["texts"]) == 0:
            st.info("Upload documents to see analytics")
        else:
            st.subheader("Document Sources")
            sources = pd.Series(brain.data["sources"]).value_counts()
            fig1 = px.pie(values=sources.values, names=sources.index, title="Documents by Source")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Ingestion Timeline")
            timeline = build_ingestion_timeline(brain.data["timestamps"])
            if timeline.empty:
                st.info("No valid ingestion timestamps are available yet.")
            else:
                fig2 = px.line(timeline, x="date", y="count", title="Documents Added Over Time")
                st.plotly_chart(fig2, use_container_width=True)

            if brain.data["tags"]:
                st.subheader("Tag Distribution")
                all_tags = [tag for tags in brain.data["tags"] for tag in tags]
                if all_tags:
                    tag_counts = pd.Series(all_tags).value_counts().head(20)
                    fig3 = px.bar(x=tag_counts.values, y=tag_counts.index, orientation="h", title="Top 20 Tags")
                    st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.header("⚙️ Settings & Export")

        st.subheader("Export Data")
        if st.button("Export Brain as JSON"):
            import json
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

        st.subheader("Danger Zone")
        if st.button("🗑️ Clear This Brain", type="secondary"):
            if st.checkbox("I understand this will delete all data in this brain"):
                brain.data = DEFAULT_BRAIN.copy()
                brain.save()
                st.success("Brain cleared successfully")
                st.rerun()

        st.divider()
        st.info(
            """
            **About Principia 2.0**
            - 100% offline operation
            - Zero API calls
            - Your data never leaves your machine
            - Open source and privacy-first
            """
        )


if __name__ == "__main__":
    main()
