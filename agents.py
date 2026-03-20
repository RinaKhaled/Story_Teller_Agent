import os
import json
import base64
from typing import Optional, List, TypedDict, Annotated, Literal

from google import genai as google_genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

image_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))


class ImageData(TypedDict):
    url: str
    caption: str


class StoryState(TypedDict):
    messages: Annotated[list, add_messages]
    prompt: str
    genre: str
    paragraphs: int
    title: Optional[str]
    story: Optional[str]
    scenes: Optional[List[str]]
    images: Optional[List[ImageData]]


@tool
def write_story(prompt: str, genre: str, paragraphs: int) -> dict:
    """Write a story and extract scene descriptions for image generation."""
    resp = image_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            f"Write a {genre} story about: {prompt}\n\n"
            f'Return JSON only, no markdown:\n'
            f'{{"title": "...", "story": "{paragraphs} paragraphs separated by \\n\\n"}}'
        ),
    )
    data = json.loads(resp.text.strip().removeprefix("```json").removesuffix("```").strip())

    scenes_resp = image_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            f"Given this story:\n\n{data['story']}\n\n"
            "Pick 4 vivid visual moments and write each as a detailed image generation prompt.\n"
            'Return JSON only, no markdown:\n{"scenes": ["...", "...", "...", "..."]}'
        ),
    )
    scenes = json.loads(scenes_resp.text.strip().removeprefix("```json").removesuffix("```").strip())["scenes"]

    return {"title": data["title"], "story": data["story"], "scenes": scenes}


@tool
def render_images(scenes: List[str]) -> List[dict]:
    """Generate images for each scene description."""
    images = []
    for prompt in scenes:
        try:
            resp = image_client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="16:9"),
                ),
            )
            data = resp.candidates[0].content.parts[0].inline_data.data
            b64 = base64.b64encode(data).decode()
            images.append({"url": f"data:image/png;base64,{b64}", "caption": prompt[:80]})
        except Exception as e:
            print(f"[render_images] {prompt[:40]!r}: {e}")

    return images


llm_with_tools = llm.bind_tools([write_story, render_images])


def orchestrator(state: StoryState) -> StoryState:
    system = SystemMessage(content=(
        "You are a story pipeline orchestrator. Given a user request, "
        "you must first call write_story to generate the story, "
        "then call render_images with the returned scenes to generate images. "
        "Always call both tools in sequence."
    ))
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


def route(state: StoryState) -> Literal["story_writer", "image_gen", "__end__"]:
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return "__end__"
    tool_name = last.tool_calls[0]["name"]
    if tool_name == "write_story":
        return "story_writer"
    if tool_name == "render_images":
        return "image_gen"
    return "__end__"


graph = StateGraph(StoryState)
graph.add_node("orchestrator", orchestrator)
graph.add_node("story_writer", ToolNode([write_story]))
graph.add_node("image_gen", ToolNode([render_images]))
graph.set_entry_point("orchestrator")
graph.add_conditional_edges("orchestrator", route)
graph.add_edge("story_writer", "orchestrator")
graph.add_edge("image_gen", "orchestrator")
pipeline = graph.compile()


NODE_LABELS = {
    "orchestrator": "Orchestrating...",
    "story_writer": "Writing story...",
    "image_gen": "Generating images...",
}


def orchestra(prompt, genre, paragraphs, status=None) -> tuple[StoryState, Optional[str]]:
    try:
        state = None
        initial_message = HumanMessage(
            content=f"Write a {genre} story about: {prompt}. Use {paragraphs} paragraphs."
        )
        for event in pipeline.stream({
            "messages": [initial_message],
            "prompt": prompt,
            "genre": genre,
            "paragraphs": paragraphs,
        }):
            node = next(iter(event))
            if status:
                status.write(NODE_LABELS.get(node, node))
            state = event[node]
    except Exception as e:
        return {}, str(e)

    if status:
        status.update(label="Done", state="complete")

    return state, None
