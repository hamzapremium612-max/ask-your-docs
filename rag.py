import os
import math
import json
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()


# The key lives in a different place depending on WHERE this code is running.
# On Streamlit Community Cloud it is in st.secrets (pasted into the dashboard).
# On this laptop it is in the .env file. Try the cloud first, fall back to local
# - so the exact same file works in both places with no editing.
# Every branch below PRINTS what it did. A silent except here means a failed
# deploy looks identical to a missing key - we would be guessing. print() goes
# straight to the Streamlit Cloud log panel. The key itself is never printed.
def get_secret(name):
    try:
        import streamlit as st
        available = list(st.secrets.keys())
        print("get_secret: st.secrets is readable. Keys present:", available)
        value = st.secrets[name]
        print("get_secret: FOUND", name, "in st.secrets (length", len(value), ")")
        return value
    except Exception as error:
        print("get_secret: st.secrets failed ->", type(error).__name__, ":", error)

    value = os.getenv(name)
    print("get_secret: os.getenv fallback ->", "FOUND" if value else "NOT FOUND")
    return value


gemini_key = get_secret("GEMINI_API_KEY")

# Fail here, with a useful sentence, instead of three frames deeper inside the
# OpenAI library saying "Missing credentials".
if not gemini_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found.\n"
        "  Locally: put it in a .env file next to this one.\n"
        "  On Streamlit Cloud: Settings -> Secrets, in TOML format:\n"
        '      GEMINI_API_KEY = "your-key-here"'
    )

# Failures get recorded here so we are never blind to what broke.
logging.basicConfig(
    filename="rag.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

ai = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=gemini_key,
)


# Some failures are worth retrying (a blip). A used-up daily quota is not -
# retrying just burns more of it. Tell them apart.
def is_quota_error(error):
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or "exceeded your current quota" in text


# --- Carried over from Project 2: read .txt or .pdf, return text ---
def read_document(path):
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text = text + page.extract_text()
        return text
    else:
        with open(path) as f:
            return f.read()


# --- Step 1: split a document into chunks ---
def chunk_text(text, max_chars=600):
    paragraphs = text.split("\n\n")      # a blank line = a paragraph break
    chunks = []
    current = ""                         # the chunk we are currently filling

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current and len(current) + len(para) > max_chars:
            chunks.append(current)       # this chunk is full - bank it
            current = para               # start a fresh one
        elif current:
            current = current + "\n\n" + para
        else:
            current = para

    if current:
        chunks.append(current)
    return chunks


# --- Step 2: turn a piece of text into numbers (its meaning fingerprint) ---
def embed(text):
    for attempt in range(1, 4):                  # up to 3 tries
        try:
            reply = ai.embeddings.create(
                input=text,
                model="gemini-embedding-001",
            )
            return reply.data[0].embedding       # a long list of numbers
        except Exception as error:
            logging.warning("embed attempt " + str(attempt) + " failed: " + str(error))
            if is_quota_error(error):            # retrying a daily quota is pointless
                logging.error("quota exhausted - not retrying")
                raise RuntimeError("Daily free-tier quota reached. See rag.log.")
            time.sleep(attempt * 3)              # back off: 3s, then 6s, then 9s

    # No usable vector means nothing downstream can work - stop loudly.
    logging.error("embedding failed after 3 attempts")
    raise RuntimeError("Embedding failed after 3 attempts. Check rag.log.")


# --- Step 3a: how aligned are two arrows? (cosine similarity) ---
def similarity(a, b):
    dot = 0          # how much the two arrows agree
    size_a = 0       # how long arrow a is (squared, for now)
    size_b = 0       # how long arrow b is (squared, for now)

    for i in range(len(a)):
        dot = dot + a[i] * b[i]
        size_a = size_a + a[i] * a[i]
        size_b = size_b + b[i] * b[i]

    # divide by the lengths so only DIRECTION is left
    return dot / (math.sqrt(size_a) * math.sqrt(size_b))


# a tiny helper so sort() knows which field to rank by
def get_score(item):
    return item["score"]


# --- Step 3b: find the chunks closest to a question ---
def search(question, library, top_n=3):
    q_vector = embed(question)          # the question becomes numbers too

    scored = []
    for entry in library:
        score = similarity(q_vector, entry["embedding"])
        scored.append({
            "score": score,
            "text": entry["text"],
            "source": entry["source"],           # carry the document name through
        })

    scored.sort(key=get_score, reverse=True)   # best score first
    return scored[:top_n]                      # keep only the top few


# --- Step 4: hand the retrieved chunks to the AI, grounded ---
def answer(question, matches):
    # stitch the winning chunks into one block of context
    context = ""
    for m in matches:
        # label each excerpt with its document so the AI can cite it
        context = context + "[from " + m["source"] + "]\n" + m["text"] + "\n\n---\n\n"

    instructions = """You answer questions about company documents, using ONLY the excerpts provided.

HOW TO ANSWER - follow this order every time:
1. Report everything the excerpts DO say that relates to the question, quoting
   specific numbers and terms.
2. Then state plainly what the excerpts do not cover.
3. You may compare, infer, and draw conclusions from the excerpt content.
4. Never state a fact that is not in the excerpts.
5. Each excerpt is labelled with the document it came from. Name that document
   when you use it, e.g. "According to it-policy.txt, ...".

A partial answer is a GOOD answer. Example:

  Question: "Do managers get more sick days than junior staff?"
  Good answer: "The handbook states all employees receive 10 paid sick days per
  year, with a medical certificate required beyond three consecutive days. It
  does not distinguish between managers and junior staff, so no difference in
  sick leave is specified."

  Bad answer: "That is not covered in the documents."

Only say "That is not covered in the documents." when the excerpts contain
nothing whatsoever related to the question."""

    for attempt in range(1, 4):                  # up to 3 tries
        try:
            reply = ai.chat.completions.create(
                model="gemini-3.1-flash-lite",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": "EXCERPTS:\n\n" + context + "QUESTION: " + question},
                ],
                temperature=0,
            )
            return reply.choices[0].message.content
        except Exception as error:
            logging.warning("answer attempt " + str(attempt) + " failed: " + str(error))
            if is_quota_error(error):            # retrying a daily quota is pointless
                logging.error("quota exhausted - not retrying")
                return "Daily free-tier quota reached. Try again tomorrow, or switch to a model with a higher limit."
            time.sleep(attempt * 3)              # back off: 3s, then 6s, then 9s

    # Here we CAN degrade gracefully - tell the user instead of crashing.
    logging.error("answer failed after 3 attempts for: " + question)
    return "Sorry - the AI service did not respond after 3 attempts. See rag.log."


# --- Phase A: build the library (chunk + embed) ---
LIBRARY_FILE = "library.json"


def build_library(folder):
    lib = []
    filenames = os.listdir(folder)               # every document in the folder
    print("Found", len(filenames), "documents in", folder)

    for filename in filenames:
        path = os.path.join(folder, filename)
        chunks = chunk_text(read_document(path))
        print("  " + filename + " -> " + str(len(chunks)) + " chunks, embedding...")

        for chunk in chunks:
            lib.append({
                "text": chunk,
                "source": filename,              # remember WHICH document it came from
                "embedding": embed(chunk),
            })
    return lib


def load_or_build_library(path):
    # Already saved from a previous run? Just load it - no API calls needed.
    if os.path.exists(LIBRARY_FILE):
        print("Loading saved library from", LIBRARY_FILE)
        with open(LIBRARY_FILE) as f:
            return json.load(f)

    # First run - do the work, then save it for next time.
    print("No saved library found. Building it...")
    lib = build_library(path)
    with open(LIBRARY_FILE, "w") as f:
        json.dump(lib, f)
    print("Saved to", LIBRARY_FILE, "- future runs will skip this.")
    return lib


# --- Phase B: the terminal interface ---
# Everything below runs ONLY when this file is run directly (python rag.py).
# When app.py IMPORTS this file, this block is skipped - so the web app can
# borrow the functions above without launching the terminal loop.
if __name__ == "__main__":
    library = load_or_build_library("documents")     # the whole folder

    print("\n" + "=" * 55)
    print("Ask questions about the documents. Type 'quit' to exit.")
    print("=" * 55)

    while True:
        question = input("\nQuestion: ").strip()   # wait for the user to type

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye.")
            break                                  # leave the loop

        if not question:                           # they just pressed enter
            continue                               # skip, ask again

        matches = search(question, library)        # find the relevant chunks
        print("\nRetrieved:")
        for m in matches:
            heading = m["text"].split("\n")[0]     # first line of the chunk
            print("   [" + str(round(m["score"], 3)) + "] " + m["source"] + " - " + heading)

        print("\nA:", answer(question, matches))   # hand them to the AI
