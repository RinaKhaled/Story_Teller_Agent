import os
import json
import base64
from typing import Optional, List, TypedDict, Annotated, Literal

from google import genai as google_genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
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
                ),
            )
            data = next(
                p.inline_data.data for p in resp.candidates[0].content.parts if p.inline_data
            )
            b64 = base64.b64encode(data).decode()
            images.append({"url": f"data:image/png;base64,{b64}", "caption": prompt[:80]})
        except Exception as e:
            print(f"[render_images] {prompt[:40]!r}: {e}")
    return images


llm_with_tools = llm.bind_tools([write_story, render_images], tool_choice="any")

def orchestrator(state: StoryState) -> StoryState:
    from langchain_core.messages import SystemMessage
    system = SystemMessage(content=(
        "You are a story pipeline orchestrator. "
        "You must ALWAYS respond by calling a tool, never with plain text. "
        "If the story has not been written yet, call write_story. "
        "If the story is written but images are not generated, call render_images with the scenes. "
        "If both are done, you may stop."
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