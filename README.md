# 📄 Ask Your Docs

A question-answering app that reads a folder of documents and answers **only**
from what is actually written in them. No guessing, no invented policy.

Built as a learning project — a small RAG (retrieval-augmented generation)
system written from scratch, including the similarity maths.

**Live demo:** _(add your Streamlit URL here after deploying)_

> The documents in `documents/` describe a fictional company, "Northfield
> Trading". Nothing in them is real.

## How it works

```
PREPARE (once)                     ANSWER (every question)
  read documents        ->           embed the question
  split into chunks     ->           score it against every chunk
  embed each chunk      ->           keep the top 3
  save to library.json  ->           hand those 3 to the AI, grounded
```

The AI never sees the whole document library — only the three most relevant
excerpts. That is the whole idea of RAG: **retrieve first, then generate.**

## Why the answers can be trusted

The model is instructed to answer from the supplied excerpts only, to name the
document it used, and to say plainly what the excerpts do not cover. Every
answer in the UI ships with an expander showing the exact text it came from —
so you can check it, not just believe it.

## Running it locally

```bash
pip install -r requirements.txt
```

Put your key in a `.env` file next to `app.py`:

```
GEMINI_API_KEY=your-key-here
```

Then:

```bash
python -m streamlit run app.py     # the web app
python rag.py                      # the same engine, in the terminal
```

`rag.py` is the engine; `app.py` is one interface onto it. The terminal loop is
guarded by `if __name__ == "__main__"`, so importing the engine does not start
it. Swap `app.py` for anything else and the engine is unchanged.

## Deploying it

The key is read by `get_secret()`, which checks Streamlit's secrets store first
and falls back to `.env`. Same file, both environments, no edits.

On Streamlit Community Cloud, paste this into **Advanced settings → Secrets**:

```toml
GEMINI_API_KEY = "your-key-here"
```

## Using your own documents

Drop `.txt` or `.pdf` files into `documents/`, **delete `library.json`**, and
run it again. It will re-chunk and re-embed.

⚠️ `library.json` is a cache and does not check whether the documents changed.
It ships with the repo so the deployed app starts instantly instead of
re-embedding on every cold start. Delete it whenever the documents change.

## Known limits

- No overlap between chunks — a fact split across a boundary can be missed.
- No score threshold — the top 3 are always returned, even when all 3 are weak.
- Brute-force similarity across every chunk. Fine for tens of documents,
  wrong for thousands (that is what a vector database is for).
- Whatever is in `documents/` is readable through the answers. That is the
  app's job, not a bug — so only put things there that you are happy to share.
