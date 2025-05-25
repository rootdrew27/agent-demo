# Following LangGraphs Guide: https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/
from typing import Annotated
from typing_extensions import TypedDict

import os

from dotenv import load_dotenv

from langchain_core.runnables.config import RunnableConfig

from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_tavily import TavilySearch  # type: ignore

load_dotenv()

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class State(TypedDict):
    messages: Annotated[list, add_messages]


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
    tool_node = ToolNode(tools=tools)

    llm_with_tavily = llm.bind_tools(tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tavily.invoke(state["messages"])]}

    # build the graph
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot", tools_condition, {"tools": "tools", END: END}
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    memory = MemorySaver()

    graph = graph_builder.compile(checkpointer=memory)

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

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    with open("output.txt", "w") as f:
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                events = graph.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config,
                    stream_mode="values",
                )

                for event in events:
                    f.write(str(event))
                    f.flush()
                    event["messages"][-1].pretty_print()

            except Exception as ex:
                print(ex)
                break
        f.close()