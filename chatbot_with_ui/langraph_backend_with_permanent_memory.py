from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import Annotated, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
import sqlite3


class my_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOllama(model="llama3.1:8b", temperature=0.2)

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)


def chat_node(state: my_state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(my_state)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
