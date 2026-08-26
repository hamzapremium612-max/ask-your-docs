import streamlit as st

# Borrow the engine from rag.py. Because rag.py guards its terminal loop with
# "if __name__ == '__main__'", importing it does NOT start that loop.
#
# QuotaExhausted is imported by NAME rather than matched on its message. The UI
# has to tell "the free demo is used up today" apart from "something broke",
# because those ask the visitor to do completely different things.
from rag import load_or_build_library, search, answer, QuotaExhausted

st.set_page_config(page_title="Ask Your Docs", page_icon="📄")

REPO = "https://github.com/hamzapremium612-max/ask-your-docs"

st.title("📄 Ask Your Docs")
st.caption("Answers come only from the documents in the documents/ folder.")


# This is a public demo on a free API tier, so running out is a NORMAL outcome,
# not a bug - and it must not look like one. Without this, a used-up quota
# reaches the visitor as a red Python traceback, which reads as "this app is
# broken" rather than "come back tomorrow". They close the tab and nobody ever
# finds out. Failing politely is the whole job of these two helpers.
def out_of_quota():
    st.info(
        "**This free demo has used up today's quota.**\n\n"
        "It runs on a free API tier with a daily cap, and enough people have "
        "tried it today to reach it. The cap resets tomorrow.\n\n"
        "If you would rather not wait, the code is open — clone it and run it "
        "with your own key: " + REPO,
        icon="🔋",
    )


def something_broke(error):
    st.error(
        "**Something went wrong answering that.**\n\n"
        "This is not your question's fault. The details are in the app log, "
        "and the code is at " + REPO + " if you want to see what it does.\n\n"
        "`" + type(error).__name__ + "`",
        icon="⚠️",
    )
    # The full error goes to the console, where the Streamlit log panel shows
    # it. The visitor gets a sentence; we get the stack.
    print("UNHANDLED:", type(error).__name__, ":", error)


# Streamlit re-runs this whole file on every interaction. Without this cache,
# the library would rebuild on every keystroke. @st.cache_resource means
# "run once, then reuse the result".
@st.cache_resource
def get_library():
    return load_or_build_library("documents")


# If the library cannot load there is no app at all, so say so plainly and
# stop, rather than letting every later line fail in a more confusing way.
try:
    library = get_library()
except QuotaExhausted:
    out_of_quota()
    st.stop()
except Exception as error:
    st.error(
        "**The document library could not be loaded, so the app cannot start.**\n\n"
        "`" + type(error).__name__ + "`",
        icon="⚠️",
    )
    print("LIBRARY LOAD FAILED:", type(error).__name__, ":", error)
    st.stop()

# Show what the assistant actually knows about.
sources = sorted(set(entry["source"] for entry in library))
st.sidebar.header("Loaded documents")
for s in sources:
    st.sidebar.write("- " + s)
st.sidebar.caption(str(len(library)) + " chunks indexed")

question = st.text_input("Your question:", placeholder="e.g. how many vacation days do I get?")

if question:
    matches = None
    reply = None

    with st.spinner("Searching the documents..."):
        try:
            matches = search(question, library)
            reply = answer(question, matches)
        except QuotaExhausted:
            out_of_quota()
        except Exception as error:
            something_broke(error)

    # Only render an answer if one was actually produced. Both handlers above
    # have already told the visitor what happened.
    if reply is not None:
        st.markdown("### Answer")
        st.write(reply)

        with st.expander("Show the excerpts this came from"):
            for m in matches:
                heading = m["text"].split("\n")[0]
                st.markdown("**" + m["source"] + "** - " + heading
                            + "  (score " + str(round(m["score"], 3)) + ")")
                st.text(m["text"])
                st.divider()
