import streamlit as st

# Borrow the engine from rag.py. Because rag.py guards its terminal loop with
# "if __name__ == '__main__'", importing it does NOT start that loop.
from rag import load_or_build_library, search, answer

st.set_page_config(page_title="Ask Your Docs", page_icon="📄")

st.title("📄 Ask Your Docs")
st.caption("Answers come only from the documents in the documents/ folder.")


# Streamlit re-runs this whole file on every interaction. Without this cache,
# the library would rebuild on every keystroke. @st.cache_resource means
# "run once, then reuse the result".
@st.cache_resource
def get_library():
    return load_or_build_library("documents")


library = get_library()

# Show what the assistant actually knows about.
sources = sorted(set(entry["source"] for entry in library))
st.sidebar.header("Loaded documents")
for s in sources:
    st.sidebar.write("- " + s)
st.sidebar.caption(str(len(library)) + " chunks indexed")

question = st.text_input("Your question:", placeholder="e.g. how many vacation days do I get?")

if question:
    with st.spinner("Searching the documents..."):
        matches = search(question, library)
        reply = answer(question, matches)

    st.markdown("### Answer")
    st.write(reply)

    with st.expander("Show the excerpts this came from"):
        for m in matches:
            heading = m["text"].split("\n")[0]
            st.markdown("**" + m["source"] + "** - " + heading
                        + "  (score " + str(round(m["score"], 3)) + ")")
            st.text(m["text"])
            st.divider()
