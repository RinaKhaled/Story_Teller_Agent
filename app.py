import json
import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage
from agents import pipeline, NODE_LABELS

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

    input_state = {
        "messages": [HumanMessage(content=f"Write a {genre} story about: {prompt}. Use {paragraphs} paragraphs.")],
        "prompt": prompt,
        "genre": genre,
        "paragraphs": paragraphs,
    }

    story_placeholder = st.empty()
    images_placeholder = st.empty()

    with st.status("Running...", expanded=True) as status:
        for event in pipeline.stream(input_state):
            node = next(iter(event))
            state = event[node]
            status.write(NODE_LABELS.get(node, node))

            if node == "story_writer":
                for msg in state.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        result = msg.content if isinstance(msg.content, dict) else json.loads(msg.content)
                        with story_placeholder.container():
                            st.subheader(result["title"])
                            st.markdown(result["story"])

            if node == "image_gen":
                for msg in state.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        images = msg.content if isinstance(msg.content, list) else json.loads(msg.content)
                        with images_placeholder.container():
                            st.divider()
                            cols = st.columns(len(images))
                            for col, img in zip(cols, images):
                                col.image(img["url"], caption=img["caption"], use_container_width=True)

        status.update(label="Done", state="complete")