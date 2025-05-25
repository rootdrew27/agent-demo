# Following LangGraphs Guide: https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/
from typing import Annotated
from typing_extensions import TypedDict

import os
import json

# from IPython.display import Image, display

from dotenv import load_dotenv

from langchain_core.messages import ToolMessage

from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


from langchain_tavily import TavilySearch  # type: ignore

load_dotenv()

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):  # empty list is falsy
            message = messages[-1]
        else:
            raise ValueError("No messages found in input to BasicToolNode.")
        outputs: list[ToolMessage] = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def route_tools(
    state: State | list,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


if __name__ == "__main__":
    if (
        input(
            "Warning: running this program will utilize the OPENAI, Tavily, and LangSmith APIs. If you are sure you want to proceed type 'continue': "
        )
        != "continue"
    ):
        exit()

    graph_builder = StateGraph(State)

    llm = init_chat_model("openai:gpt-4.1-mini", api_key=OPENAI_API_KEY)

    # create tools
    tool = TavilySearch(max_results=1)
    tools = [tool]
    tool_node = BasicToolNode(tools=tools)

    llm_with_tavily = llm.bind_tools(tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tavily.invoke(state["messages"])]}

    # build the graph
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot", route_tools, {"tools": "tools", END: END}
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    graph = graph_builder.compile()

    # try:
    #     display(Image(graph.get_graph().draw_mermaid_png()))
    # except Exception:
    #     pass

    def stream_graph_updates(user_input: str):
        for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}
        ):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except Exception as ex:
            print(ex)
            break
