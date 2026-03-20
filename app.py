import json
import streamlit as st
from langchain_core.messages import ToolMessage
from agents import orchestra

st.set_page_config(page_title="Story Teller", page_icon="📖")
st.title("Story Teller")

with st.sidebar:
    genre = st.selectbox("Genre", ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Horror", "Adventure"])
    paragraphs = st.slider("Paragraphs", 1, 6, 3)

prompt = st.text_area("Story idea", placeholder="A lone astronaut finds an ancient alien library on Mars...")

if st.button("Generate", type="primary"):
    if not prompt.strip():
        st.warning("Enter a story idea first.")
        st.stop()

    with st.status("Running...", expanded=True) as status:
        state, error = orchestra(prompt, genre, paragraphs, status)

    if error:
        st.error(error)
        st.stop()

    # extract tool results from messages
    story_data, images = None, None
    for msg in state.get("messages", []):
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(msg.content)
                if "story" in result:
                    story_data = result
                elif isinstance(result, list) and result and "url" in result[0]:
                    images = result
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

    if not story_data:
        st.error("No story was generated.")
        st.stop()

    st.subheader(story_data["title"])
    st.markdown(story_data["story"])

    if images:
        st.divider()
        cols = st.columns(len(images))
        for col, img in zip(cols, images):
            col.image(img["url"], caption=img["caption"], use_container_width=True)
