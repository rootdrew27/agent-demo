# Following LangGraphs Guide: https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/
from typing import Annotated
from typing_extensions import TypedDict

import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langchain.chat_models import init_chat_model

load_dotenv()


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


if __name__ == "__main__":
    graph_builder = StateGraph(State)

    openai_api_key = os.environ.get("devkey")
    llm = init_chat_model("openai:gpt-4.1-mini", api_key=openai_api_key)

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile()

    def stream_graph_updates(user_input: str):
        with open("basic-chatbot-output.txt", "a") as f:
            f.write(f"===\nInput: {user_input}\n\n")
            for mode, event in graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                stream_mode=["values", "updates"],
            ):
                if mode == "updates" and isinstance(event, dict):
                    print("Assistant:", event["messages"][-1].content)
                elif mode == "values":
                    f.write(f"Update: {event}")
                    f.flush()
                else:
                    raise ValueError(
                        "Only stream_modes handled are 'values' and 'updates'."
                    )

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
