# 📚 Assistant documentaire (RAG)

Système RAG qui permet de gérer un corpus de documents (PDF, TXT, MD) et de répondre aux questions
des utilisateurs à partir de ce corpus, avec citation des sources.

Streamlit · LangChain · Groq · recherche sémantique par embeddings.

## Installation

Python 3.10+ requis.

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Windows : copy .env.example .env
```

Ouvrez `.env` et renseignez votre clé Groq (gratuite : <https://console.groq.com/keys>) :

```
GROQ_API_KEY=votre_clé_ici
```

> Au premier lancement, le modèle d'embeddings (~470 Mo) est téléchargé : prévoyez une connexion Internet.
> Pour s'en passer, ajoutez `RETRIEVAL=bm25` dans `.env` (recherche par mots-clés).

## Lancement

```bash
streamlit run app.py
```

## Utilisation

1. Ajoutez des documents avec le bouton 📎 de la zone de saisie, ou placez-les dans `corpus/` puis cliquez sur **Synchroniser**.
2. Posez votre question : la réponse cite les passages utilisés (`[1]`, `[2]`…), visibles dans **Sources**.
3. Supprimez un document avec 🗑 dans la barre latérale.

Si la réponse n'est pas dans le corpus, l'assistant le dit.

## Fonctionnement

Documents → découpage → embeddings → recherche des passages proches de la question →
évaluation par le LLM (reformulation et nouvelle recherche si insuffisant) → réponse avec sources.

## Structure

```
app.py            # Interface Streamlit
rag_core.py       # Noyau RAG
requirements.txt
.env.example
corpus/           # Documents
index/            # Index généré
```
