from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class WorkflowInput(TypedDict):
    total_marks: int
    total_subjects: int
    total_percentage: float


def calculate_percentage(state: WorkflowInput) -> float:
    marks = state["total_marks"]
    subjects = state["total_subjects"]
    percentage = (marks / subjects) * 100
    state["total_percentage"] = percentage
    return state


graph = StateGraph(WorkflowInput)
graph.add_node(calculate_percentage)
graph.add_edge(START, "calculate_percentage")
graph.add_edge("calculate_percentage", END)
result = graph.compile()

initial_state = {
    "total_marks": 450,
    "total_subjects": 5,
}

final_state = result.invoke(initial_state)

png = result.get_graph().draw_mermaid_png()

with open("workflow.png", "wb") as f:
    f.write(png)

print("Workflow saved as workflow.png")
