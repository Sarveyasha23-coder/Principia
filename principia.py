"""Principia: local-first personal knowledge workspace.
Run: pip install -r requirements.txt && streamlit run principia.py
"""
from __future__ import annotations
import hashlib, io, json, re, sqlite3
from datetime import datetime, date
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "principia.sqlite3"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE, OVERLAP, MAX_MB = 240, 40, 40

st.set_page_config(page_title="Principia — Think clearly", page_icon="🧠", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp{background:radial-gradient(ellipse at 10% 0%,#20274b 0%,#11172b 45%,#090e1b 100%);color:#f3f5ff;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:rgba(7,11,23,.94);border-right:1px solid #ffffff18}
.block-container{max-width:1480px;padding-top:1.3rem}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em}
.hero{padding:1.8rem 2rem;border:1px solid #a99aff44;border-radius:24px;background:linear-gradient(115deg,#4f46e538,#8b5cf61c,#ffffff06);margin-bottom:1.2rem}
.hero small{color:#c9c0ff;letter-spacing:.16em;text-transform:uppercase;font-weight:700}
.hero h1{font-size:clamp(2.4rem,4vw,3.4rem);margin:.35rem 0;color:white}
.hero p{color:#c5cce0;font-size:1rem;margin:0;max-width:800px}
div[data-testid="stMetric"]{background:#ffffff08;border:1px solid #ffffff15;padding:14px;border-radius:14px}
.stButton button,.stDownloadButton button{border-radius:10px;border:1px solid #a99aff44;font-weight:650}
.stButton button[kind="primary"]{background:linear-gradient(100deg,#5b4be5,#8b5cf6);color:white;border:0}
.stTextInput input,.stTextArea textarea{background:#ffffff0b;color:white;border-color:#ffffff20;border-radius:10px}
div[data-testid="stExpander"]{border:1px solid #ffffff18;border-radius:12px}
</style>""", unsafe_allow_html=True)

# Database and schema
def connect():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("PRAGMA journal_mode=WAL")
    return c

def init_db():
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS workspaces(name TEXT PRIMARY KEY, created_at TEXT NOT NULL, purpose TEXT NOT NULL DEFAULT 'General');
        CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY, workspace TEXT NOT NULL, title TEXT NOT NULL, digest TEXT NOT NULL, added_at TEXT NOT NULL, tags TEXT NOT NULL, chars INTEGER NOT NULL, pages INTEGER, UNIQUE(workspace,digest), FOREIGN KEY(workspace) REFERENCES workspaces(name) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL, workspace TEXT NOT NULL, part INTEGER NOT NULL, content TEXT NOT NULL, embedding BLOB, dim INTEGER, FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS chunks_ws ON chunks(workspace);
        CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, workspace TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL, tags TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, FOREIGN KEY(workspace) REFERENCES workspaces(name) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY, workspace TEXT NOT NULL, title TEXT NOT NULL, details TEXT NOT NULL DEFAULT '', due TEXT, priority TEXT NOT NULL DEFAULT 'Normal', status TEXT NOT NULL DEFAULT 'Open', created_at TEXT NOT NULL, FOREIGN KEY(workspace) REFERENCES workspaces(name) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS activity(id INTEGER PRIMARY KEY, workspace TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL);
        """)
        c.execute("INSERT OR IGNORE INTO workspaces VALUES(?,?,?)",("My Knowledge",datetime.now().isoformat(timespec="seconds"),"Personal"))

def log(ws, action, detail):
    with connect() as c:
        c.execute("INSERT INTO activity(workspace,action,detail,created_at) VALUES(?,?,?,?)",(ws,action,detail,datetime.now().isoformat(timespec="seconds")))

def clean_name(s):
    s = re.sub(r"[^a-zA-Z0-9 _-]", "", s).strip()
    s = re.sub(r"\s+", "-", s)[:48]
    if not s: raise ValueError("Use letters, numbers, spaces, hyphens, or underscores.")
    return s

@st.cache_resource(show_spinner="Loading local embedding model…")
def model():
    return SentenceTransformer(MODEL_NAME)

def parse_file(upload):
    raw, name = upload.getvalue(), upload.name.lower()
    if len(raw) > MAX_MB*1024*1024: raise ValueError(f"File exceeds {MAX_MB} MB.")
    if name.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(raw), strict=False)
        return "\n\n".join(p.extract_text() or "" for p in pdf.pages).strip(), len(pdf.pages)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(raw))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows: parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n\n".join(parts).strip(), None
    return raw.decode("utf-8-sig", errors="replace").strip(), None

def chunk_text(text):
    words = re.findall(r"\S+", text)
    chunks, start = [], 0
    while start < len(words):
        end = min(start+CHUNK_SIZE,len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk: chunks.append(chunk)
        if end >= len(words): break
        start = max(end-OVERLAP,start+1)
    return chunks

def ingest(ws, upload, tags, encoder):
    text, pages = parse_file(upload)
    if not text: return "empty", 0
    chunks = chunk_text(text)
    digest = hashlib.sha256(upload.getvalue()).hexdigest()
    with connect() as c:
        if c.execute("SELECT 1 FROM documents WHERE workspace=? AND digest=?",(ws,digest)).fetchone(): return "duplicate",0
        doc_id = c.execute("INSERT INTO documents(workspace,title,digest,added_at,tags,chars,pages) VALUES(?,?,?,?,?,?,?)",
            (ws,upload.name,digest,datetime.now().isoformat(timespec="seconds"),json.dumps(tags),len(text),pages)).lastrowid
    vectors = encoder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    with connect() as c:
        c.executemany("INSERT INTO chunks(document_id,workspace,part,content,embedding,dim) VALUES(?,?,?,?,?,?)",
            [(doc_id,ws,i,t,np.asarray(v,dtype=np.float32).tobytes(),len(v)) for i,(t,v) in enumerate(zip(chunks,vectors))])
    log(ws,"Document indexed",upload.name)
    return "added",len(chunks)

def workspace_rows(ws):
    with connect() as c:
        docs=[dict(r) for r in c.execute("SELECT * FROM documents WHERE workspace=? ORDER BY added_at DESC",(ws,))]
        chunks=[dict(r) for r in c.execute("SELECT ch.*,d.title,d.added_at AS doc_date,d.tags,d.pages FROM chunks ch JOIN documents d ON d.id=ch.document_id WHERE ch.workspace=? ORDER BY ch.id",(ws,))]
        notes=[dict(r) for r in c.execute("SELECT * FROM notes WHERE workspace=? ORDER BY updated_at DESC",(ws,))]
        tasks=[dict(r) for r in c.execute("SELECT * FROM tasks WHERE workspace=? ORDER BY CASE priority WHEN 'High' THEN 0 WHEN 'Normal' THEN 1 ELSE 2 END,due",(ws,))]
        activity=[dict(r) for r in c.execute("SELECT * FROM activity WHERE workspace=? ORDER BY id DESC LIMIT 20",(ws,))]
    return docs,chunks,notes,tasks,activity

def search(query, ws, encoder, mode="Hybrid", limit=5, tag="All tags"):
    _, rows, _, _, _ = workspace_rows(ws)
    if tag != "All tags": rows=[r for r in rows if tag in json.loads(r["tags"] or "[]")]
    if not query.strip() or not rows: return []
    scores=np.zeros(len(rows),dtype=float)
    if mode in ("Hybrid","Semantic"):
        q=np.asarray(encoder.encode([query],normalize_embeddings=True)[0],dtype=np.float32)
        for i,r in enumerate(rows):
            if r["embedding"] and r["dim"]==len(q):
                v=np.frombuffer(r["embedding"],dtype=np.float32)
                if len(v)==len(q) and np.isfinite(v).all(): scores[i]=float(np.dot(q,v))
        if mode=="Hybrid": scores*=.72
    if mode in ("Hybrid","Keyword"):
        try:
            tf=TfidfVectorizer(ngram_range=(1,2),stop_words="english",sublinear_tf=True)
            mat=tf.fit_transform([r["content"] for r in rows])
            kw=cosine_similarity(tf.transform([query]),mat)[0]
        except ValueError: kw=np.zeros(len(rows))
        scores=kw if mode=="Keyword" else scores+.28*kw
    out=[]
    for i in np.argsort(scores)[::-1][:limit]:
        if scores[i] > 0:
            item=rows[int(i)].copy(); item["score"]=float(scores[i]); out.append(item)
    return out

def evidence_answer(question, results):
    if not results: return "I couldn't find supporting passages. Try different wording, remove the tag filter, or add a relevant document."
    terms=set(re.findall(r"\b[a-zA-Z0-9]{3,}\b",question.lower()))
    pool=[]
    for r in results[:5]:
        for s in re.split(r"(?<=[.!?])\s+",r["content"]):
            s=s.strip()
            if len(s)<24: continue
            st_terms=set(re.findall(r"\b[a-zA-Z0-9]{3,}\b",s.lower()))
            score=len(terms&st_terms)/max(1,len(terms))+min(len(s),400)/4000
            pool.append((score,s,r))
    pool.sort(key=lambda x:x[0],reverse=True)
    chosen=[]; seen=set()
    for _,s,r in pool:
        if s.lower() not in seen: chosen.append((s,r)); seen.add(s.lower())
        if len(chosen)==3: break
    if not chosen: chosen=[(r["content"][:400],r) for r in results[:2]]
    lines=["**Relevant evidence from your workspace**"]
    for s,r in chosen: lines.append(f"- “{s}”\n\n  **Source:** {r['title']} · relevance `{r['score']:.3f}`")
    lines.append("*These are extracted passages, not fabricated answers from a generative AI model.*")
    return "\n\n".join(lines)

def export_json(ws):
    docs,chunks,notes,tasks,_=workspace_rows(ws)
    for d in docs: d.pop("id",None)
    clean_chunks=[{"document":r["title"],"part":r["part"],"content":r["content"]} for r in chunks]
    clean_notes=[{k:n[k] for k in ("title","content","tags","created_at","updated_at")} for n in notes]
    clean_tasks=[{k:t[k] for k in ("title","details","due","priority","status","created_at")} for t in tasks]
    return json.dumps({"product":"Principia","schema_version":1,"workspace":ws,"exported_at":datetime.now().isoformat(timespec="seconds"),"documents":docs,"chunks":clean_chunks,"notes":clean_notes,"tasks":clean_tasks},ensure_ascii=False,indent=2).encode()

def main():
    init_db()
    for key,default in [("ws","My Knowledge"),("chats",{}),("last",{})]:
        if key not in st.session_state: st.session_state[key]=default
    with st.sidebar:
        st.markdown("## 🧠 Principia")
        st.caption("Ideas → knowledge → action")
        with connect() as c: names=[r["name"] for r in c.execute("SELECT name FROM workspaces ORDER BY name")]
        if st.session_state.ws not in names: st.session_state.ws=names[0]
        ws=st.selectbox("Workspace",names,index=names.index(st.session_state.ws))
        st.session_state.ws=ws
        with st.expander("＋ New workspace"):
            n=st.text_input("Name",placeholder="School, Team, Personal")
            p=st.selectbox("Purpose",["Personal","School","Family","Work","Research","General"])
            if st.button("Create workspace",use_container_width=True):
                try:
                    name=clean_name(n)
                    with connect() as c: c.execute("INSERT INTO workspaces VALUES(?,?,?)",(name,datetime.now().isoformat(timespec="seconds"),p))
                    log(name,"Workspace created",p); st.session_state.ws=name; st.rerun()
                except (ValueError,sqlite3.IntegrityError) as e: st.error(str(e))
        st.divider()
        st.markdown("### Add knowledge")
        files=st.file_uploader("PDF, DOCX, TXT, Markdown",type=["pdf","docx","txt","md"],accept_multiple_files=True)
        tags_raw=st.text_input("Tags",placeholder="study, project, reference")
        tags=list(dict.fromkeys(x.strip() for x in tags_raw.split(",") if x.strip()))
        if st.button("Index documents",type="primary",use_container_width=True):
            if not files: st.warning("Choose at least one file.")
            else:
                try:
                    enc=model(); progress=st.progress(0); added=dupes=empty=failed=0
                    for i,f in enumerate(files):
                        try:
                            result,_=ingest(ws,f,tags,enc)
                            if result=="added": added+=1
                            elif result=="duplicate": dupes+=1
                            else: empty+=1
                        except Exception as e: failed+=1; st.warning(f"{f.name}: {e}")
                        progress.progress((i+1)/len(files))
                    st.success(f"Added {added}; duplicates {dupes}; empty {empty}; failed {failed}."); st.rerun()
                except Exception as e: st.error(f"Indexing failed: {e}")
        st.caption("Local SQLite storage. The embedding model may download once.")
        st.divider()
        st.markdown("### Quick capture")
        with st.form("capture",clear_on_submit=True):
            nt=st.text_input("Note title",placeholder="An idea worth keeping")
            nc=st.text_area("Note",placeholder="Capture a thought, decision, lesson…",height=90)
            ng=st.text_input("Tags",placeholder="idea, learning")
            save=st.form_submit_button("Save note",use_container_width=True)
        if save:
            if not nt.strip() or not nc.strip(): st.warning("Add a title and note content.")
            else:
                now=datetime.now().isoformat(timespec="seconds")
                with connect() as c: c.execute("INSERT INTO notes(workspace,title,content,tags,created_at,updated_at) VALUES(?,?,?,?,?,?)",(ws,nt.strip(),nc.strip(),json.dumps([x.strip() for x in ng.split(",") if x.strip()]),now,now))
                log(ws,"Note captured",nt.strip()); st.success("Saved."); st.rerun()

    docs,chunks,notes,tasks,activity=workspace_rows(ws)
    alltags=set()
    for r in docs+notes:
        try: alltags.update(json.loads(r["tags"] or "[]"))
        except (ValueError,TypeError): pass
    st.markdown(f"""<div class="hero"><small>PERSONAL KNOWLEDGE OS · {ws}</small><h1>Think clearly. <span style="color:#a99aff">Find anything.</span></h1><p>One calm workspace for learning, family life, research, and meaningful work. Keep the evidence close and the next action clear.</p></div>""",unsafe_allow_html=True)
    a,b,c,d,e=st.columns(5)
    a.metric("Documents",len(docs)); b.metric("Indexed passages",len(chunks)); c.metric("Notes",len(notes)); d.metric("Open tasks",sum(t["status"]!="Done" for t in tasks)); e.metric("Tags",len(alltags))
    home,ask,library,notes_tab,tasks_tab,insights,settings=st.tabs(["⌂ Overview","✦ Ask & Search","▤ Library","✎ Notes","✓ Tasks","◫ Insights","⚙ Settings"])

    with home:
        l,r=st.columns([1.2,1])
        with l:
            st.subheader("Your next move")
            opened=[t for t in tasks if t["status"]!="Done"]
            if opened:
                for t in opened[:5]:
                    st.markdown(f"{'🔴' if t['priority']=='High' else '🟡' if t['priority']=='Normal' else '⚪'} **{t['title']}** · {t['status']}")
                    st.caption(f"Due: {t['due'] or 'not set'} · {t['priority']}")
            else: st.info("No open tasks. Add a small, specific next action in Tasks.")
            st.markdown("#### Suggested ways to use Principia")
            st.markdown("- **Students:** upload readings, then search for concepts and examples.\n- **Families:** organize shared reference material and checklists.\n- **Professionals:** collect briefs, meeting notes, research, and next actions.")
        with r:
            st.subheader("Recently added")
            if not docs and not notes: st.info("Your workspace is ready. Upload a file or save a note from the sidebar.")
            for x in docs[:4]: st.markdown(f"📄 **{x['title']}**"); st.caption(f"{x['added_at'][:16]} · {x['chars']:,} characters")
            for x in notes[:3]: st.markdown(f"📝 **{x['title']}**"); st.caption(f"Updated {x['updated_at'][:16]}")
        st.divider(); st.subheader("Recent activity")
        if activity: st.dataframe(pd.DataFrame(activity)[["created_at","action","detail"]],use_container_width=True,hide_index=True)
        else: st.caption("Your activity appears here as you work.")

    with ask:
        st.subheader("Ask your knowledge")
        st.caption("Search by meaning or exact wording. Review the original passage before relying on it.")
        tagopts=["All tags"]+sorted(alltags)
        tag=st.selectbox("Filter by tag",tagopts,key=f"tag_{ws}")
        with st.form("searchform"):
            q=st.text_input("What are you trying to find?",placeholder="What are the key ideas in my notes about energy?")
            mode=st.selectbox("Search strategy",["Hybrid","Semantic","Keyword"])
            limit=st.slider("Passages",1,10,5)
            go=st.form_submit_button("Search workspace",type="primary",use_container_width=True)
        if go:
            if not q.strip(): st.warning("Enter a question or keywords.")
            elif not chunks: st.info("Add a document first.")
            else:
                try:
                    with st.spinner("Searching locally…"): found=search(q,ws,model(),mode,limit,tag)
                    st.session_state.last[ws]={"q":q,"mode":mode,"found":found}
                except Exception as e: st.error(f"Search failed: {e}")
        last=st.session_state.last.get(ws)
        if last:
            st.markdown(f"#### {len(last['found'])} passage(s) · {last['mode']}")
            for i,x in enumerate(last["found"],1):
                with st.expander(f"{i}. {x['title']} · relevance {x['score']:.3f}",expanded=i==1):
                    st.caption(f"Added {x['doc_date'][:16]} · tags: {', '.join(json.loads(x['tags'] or '[]')) or 'none'}")
                    st.write(x["content"])
            st.markdown("#### Evidence summary"); st.markdown(evidence_answer(last["q"],last["found"]))
        st.divider(); st.subheader("Chat with your library")
        history=st.session_state.chats.setdefault(ws,[])
        for msg in history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        with st.form("chatform",clear_on_submit=True):
            prompt=st.text_input("Ask a question",placeholder="Compare the main ideas across my documents…")
            send=st.form_submit_button("Ask Principia",type="primary")
        if send:
            if not prompt.strip(): st.warning("Type a question first.")
            elif not chunks: st.info("Add documents first.")
            else:
                history.append({"role":"user","content":prompt.strip()})
                try: response=evidence_answer(prompt,search(prompt,ws,model(),"Hybrid",5))
                except Exception as ex: response=f"Search failed: {ex}"
                history.append({"role":"assistant","content":response}); st.rerun()
        if history and st.button("Clear chat for this workspace"): st.session_state.chats[ws]=[]; st.rerun()

    with library:
        st.subheader("Document library")
        if not docs: st.info("No documents yet. Upload PDF, Word, TXT, or Markdown files in the sidebar.")
        else:
            view=[{"Title":x["title"],"Added":x["added_at"],"Characters":x["chars"],"Pages":x["pages"] or "—","Tags":", ".join(json.loads(x["tags"] or "[]"))} for x in docs]
            st.dataframe(pd.DataFrame(view),use_container_width=True,hide_index=True)
            st.caption("Identical file content is only indexed once per workspace.")
            with st.expander("Delete a document and its indexed passages"):
                title=st.selectbox("Document title",list(dict.fromkeys(x["title"] for x in docs)))
                confirm=st.checkbox("I understand this also deletes its search passages.",key=f"del_doc_{ws}_{title}")
                if st.button("Delete document",disabled=not confirm):
                    with connect() as c: c.execute("DELETE FROM documents WHERE workspace=? AND title=?",(ws,title))
                    log(ws,"Document deleted",title); st.rerun()

    with notes_tab:
        st.subheader("Notes and ideas")
        if not notes: st.info("Use Quick capture in the sidebar to save your first note.")
        for n in notes:
            with st.expander(f"{n['title']} · {n['updated_at'][:16]}"):
                with st.form(f"editnote_{n['id']}"):
                    et=st.text_input("Title",n["title"]); ec=st.text_area("Content",n["content"],height=130)
                    eg=st.text_input("Tags",", ".join(json.loads(n["tags"] or "[]")))
                    update=st.form_submit_button("Save changes")
                if update:
                    with connect() as c: c.execute("UPDATE notes SET title=?,content=?,tags=?,updated_at=? WHERE id=? AND workspace=?",(et.strip(),ec.strip(),json.dumps([x.strip() for x in eg.split(",") if x.strip()]),datetime.now().isoformat(timespec="seconds"),n["id"],ws))
                    log(ws,"Note edited",et.strip()); st.rerun()
                if st.button("Delete note",key=f"del_note_{n['id']}"):
                    with connect() as c: c.execute("DELETE FROM notes WHERE id=? AND workspace=?",(n["id"],ws))
                    log(ws,"Note deleted",n["title"]); st.rerun()

    with tasks_tab:
        st.subheader("Tasks and next actions")
        with st.form("newtask",clear_on_submit=True):
            title=st.text_input("Task",placeholder="Review chapter 3")
            detail=st.text_area("Details (optional)",height=70)
            x,y=st.columns(2); due=x.date_input("Due date",value=None); priority=y.selectbox("Priority",["Normal","High","Low"])
            add=st.form_submit_button("Add task",type="primary",use_container_width=True)
        if add:
            if not title.strip(): st.warning("Give the task a title.")
            else:
                with connect() as c: c.execute("INSERT INTO tasks(workspace,title,details,due,priority,status,created_at) VALUES(?,?,?,?,?,'Open',?)",(ws,title.strip(),detail.strip(),due.isoformat() if isinstance(due,date) else None,priority,datetime.now().isoformat(timespec="seconds")))
                log(ws,"Task added",title.strip()); st.rerun()
        if not tasks: st.info("No tasks yet.")
        for t in tasks:
            with st.container(border=True):
                x,y,z=st.columns([4,1.4,1])
                x.markdown(f"**{t['title']}**"); x.caption(f"{t['details'] or 'No details'} · {t['priority']} priority · due {t['due'] or 'not set'}")
                status=y.selectbox("Status",["Open","In progress","Done"],index=["Open","In progress","Done"].index(t["status"]),key=f"status_{t['id']}",label_visibility="collapsed")
                if status!=t["status"]:
                    with connect() as c: c.execute("UPDATE tasks SET status=? WHERE id=? AND workspace=?",(status,t["id"],ws))
                    log(ws,"Task updated",f"{t['title']}: {status}"); st.rerun()
                if z.button("Delete",key=f"taskdel_{t['id']}"):
                    with connect() as c: c.execute("DELETE FROM tasks WHERE id=? AND workspace=?",(t["id"],ws))
                    log(ws,"Task deleted",t["title"]); st.rerun()

    with insights:
        st.subheader("Knowledge insights")
        if not docs and not notes: st.info("Add documents or notes to see insights.")
        else:
            x,y=st.columns(2)
            if docs:
                df=pd.DataFrame(docs)
                with x:
                    fig=px.bar(df.groupby("title",as_index=False).size(),x="title",y="size",title="Documents in workspace",labels={"title":"Document","size":"Count"})
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaff")
                    st.plotly_chart(fig,use_container_width=True)
                with y:
                    df["day"]=pd.to_datetime(df["added_at"],errors="coerce").dt.date
                    tl=df.dropna(subset=["day"]).groupby("day").size().reset_index(name="Documents")
                    if not tl.empty:
                        fig=px.line(tl,x="day",y="Documents",markers=True,title="Knowledge added over time")
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaff")
                        st.plotly_chart(fig,use_container_width=True)
            tags_all=[]
            for row in docs+notes:
                try: tags_all.extend(json.loads(row["tags"] or "[]"))
                except (ValueError,TypeError): pass
            if tags_all:
                tdf=pd.Series(tags_all).value_counts().head(15).rename_axis("Tag").reset_index(name="Count")
                fig=px.bar(tdf,x="Count",y="Tag",orientation="h",title="Most-used tags")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaff")
                st.plotly_chart(fig,use_container_width=True)

    with settings:
        st.subheader("Portability and workspace settings")
        st.download_button("Export workspace JSON",data=export_json(ws),file_name=f"principia-{re.sub(r'[^a-zA-Z0-9_-]','-',ws)}.json",mime="application/json",use_container_width=True)
        if DB_PATH.exists(): st.download_button("Download SQLite backup",data=DB_PATH.read_bytes(),file_name="principia-backup.sqlite3",mime="application/octet-stream",use_container_width=True)
        st.divider()
        if ws=="My Knowledge": st.info("The default workspace is protected from deletion.")
        else:
            sure=st.checkbox(f"Permanently delete {ws} and its documents, notes, and tasks.",key=f"delete_ws_{ws}")
            if st.button("Delete workspace",disabled=not sure):
                with connect() as c:
                    c.execute("DELETE FROM workspaces WHERE name=?",(ws,)); c.execute("DELETE FROM activity WHERE workspace=?",(ws,))
                st.session_state.chats.pop(ws,None); st.session_state.last.pop(ws,None); st.session_state.ws="My Knowledge"; st.rerun()
        st.markdown("#### Capability and launch notes")
        st.markdown("- Local SQLite database stores metadata, extracted text, chunks, and embeddings.\n- Semantic search uses `all-MiniLM-L6-v2`; first launch may download the model.\n- Answers show extracted evidence; this app does not call a cloud LLM.\n- Scanned PDFs need OCR before indexing.\n- Before public/multi-user launch, add authentication, access controls, tests, monitoring, backup automation, and a deployment security review.")

if __name__ == "__main__":
    main()
