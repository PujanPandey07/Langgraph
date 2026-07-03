from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

llm = ChatOllama(model="llama3.1:8b", temperature=0.7)


class my_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: my_state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


checkpointer = MemorySaver()


graph = StateGraph(my_state)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

thread_id = "1"


def my_chatbot():
    while True:
        user_input = input("User: ")
        print("User input:", user_input)
        if user_input.strip().lower() in ["exit", "quit", "bye", "goodbye", "stop", "done"]:
            print("Exiting the chatbot.")
            break
        config = {'configurable': {'thread_id': thread_id}}
        response = chatbot.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)
        print("Chatbot:", response["messages"][-1].content)
