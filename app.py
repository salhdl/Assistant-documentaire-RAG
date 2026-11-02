import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from rag_core import GROQ_MODEL, KnowledgeBase, answer_question  # noqa: E402

st.set_page_config(page_title="Assistant documentaire (RAG)", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Chargement de la base documentaire…")
def get_kb() -> KnowledgeBase:
    return KnowledgeBase(corpus_dir="corpus", index_dir="index")


kb = get_kb()
st.session_state.setdefault("messages", [])


# ---------------- Fonctions ----------------
def new_conversation():
    st.session_state.messages = []
    st.session_state.pop("pending", None)  # évite de relancer une question en attente


def ingest_files(files) -> str:
    """Enregistre les fichiers joints dans corpus/, les indexe, et retourne un compte rendu."""
    ok, bad = [], []
    for f in files:
        try:
            kb.add_file(f.name, f.getvalue())
            ok.append(Path(f.name).name)
        except Exception as e:
            bad.append(f"{f.name} ({e})")
    res = kb.sync() if ok else {"added": [], "updated": []}
    passages = {r["document"]: r["passages"] for r in kb.stats()}
    lines = []
    for name in ok:
        status = ("ajouté" if name in res["added"] else
                  "mis à jour" if name in res["updated"] else "déjà indexé")
        n = passages.get(name, 0)
        lines.append(f"✅ **{name}** {status} ({n} passages)" if n else
                     f"⚠️ **{name}** : aucun texte extractible (PDF scanné ou fichier vide ?)")
    lines += [f"⚠️ Ignoré : {b}" for b in bad]
    return "\n\n".join(lines)


def show_sources(sources: list[dict]):
    with st.expander(f"Sources ({len(sources)})"):
        for n, s in enumerate(sources, 1):
            page = f" · p. {s['page']}" if s["page"] else ""
            st.markdown(f"**[{n}] {s['source']}{page}** — score {s['score']:.2f}")
            st.caption(s["text"])


# ---------------- Barre latérale : gestion du corpus ----------------
with st.sidebar:
    st.header("📁 Corpus")
    if kb.files() and st.button("🔄 Synchroniser le dossier corpus/", width="stretch"):
        with st.spinner("Synchronisation…"):
            res = kb.sync()
        st.toast(f"{len(res['added'])} ajouté(s), {len(res['updated'])} mis à jour, "
                 f"{len(res['removed'])} retiré(s)")
        st.rerun()

    st.subheader("Documents indexés")
    stats = kb.stats()
    if not stats:
        st.info("Aucun document indexé. Joignez un fichier avec le bouton 📎 de la zone de saisie "
                "(en bas), ou placez-le dans `corpus/` puis synchronisez.")
    for row in stats:
        c1, c2 = st.columns([4, 1])
        c1.caption(f"**{row['document']}** — {row['passages']} passages")
        if c2.button("🗑", key=f"del_{row['document']}", help="Supprimer ce document"):
            kb.remove(row["document"])
            st.rerun()

    st.divider()
    with st.expander("⚙️ Options avancées"):
        k = st.slider("Passages lus par question (k)", 1, 10, 5,
                      help="Nombre de passages du corpus donnés au modèle pour répondre. "
                           "5 convient dans la plupart des cas.")
    if os.getenv("GROQ_API_KEY"):
        st.caption(f"Génération : Groq · `{GROQ_MODEL}`")
    else:
        st.warning("`GROQ_API_KEY` absente : mode extractif (sans réponse rédigée).")


# ---------------- Conversation ----------------
col_titre, col_bouton = st.columns([5, 1])
with col_titre:
    st.title("📚 Assistant documentaire")
    st.caption("Posez une question, ou joignez un document avec 📎 : "
               "les réponses s'appuient uniquement sur votre corpus.")
with col_bouton:
    st.button("➕ Nouvelle conversation", on_click=new_conversation, width="stretch")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

prompt = st.chat_input("Posez votre question ou joignez un document (📎)…",
                       accept_file="multiple", file_type=["pdf", "txt", "md"])

question = None
if prompt:
    files = list(getattr(prompt, "files", None) or [])
    text = (getattr(prompt, "text", None) or "").strip()
    if files:
        names = ", ".join(f.name for f in files)
        st.session_state.messages.append({"role": "user", "content": f"📎 {names}"})
        with st.chat_message("user"):
            st.markdown(f"📎 {names}")
        with st.chat_message("assistant"), st.spinner("Indexation en cours…"):
            note = ingest_files(files)
        st.session_state.messages.append({"role": "assistant", "content": note,
                                          "sources": [], "steps": []})
        if text:
            st.session_state.pending = text   # la question sera traitée après le rafraîchissement
        st.rerun()                            # met à jour la liste des documents
    question = text or None
if question is None:
    question = st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not kb.n_chunks:
            answer, sources, steps = ("Le corpus est vide : joignez d'abord un document avec 📎.", [], [])
            st.markdown(answer)
        else:
            t0 = time.perf_counter()
            with st.spinner("Recherche, évaluation et génération…"):
                try:
                    history = [{"role": m["role"], "content": m["content"]}
                               for m in st.session_state.messages[:-1]]
                    res = answer_question(kb, question, k, history)
                    answer, sources, steps = res["answer"], res["sources"], res["steps"]
                except Exception as e:  # clé invalide, réseau, quota...
                    sources, steps, answer = [], [], f"⚠️ Erreur : {e}"
            st.markdown(answer)
            if sources:
                show_sources(sources)
            st.caption(f"⏱ {time.perf_counter() - t0:.1f} s")
    st.session_state.messages.append({"role": "assistant", "content": answer,
                                      "sources": sources, "steps": steps})