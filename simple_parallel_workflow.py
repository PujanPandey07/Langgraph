from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class my_state(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int

    strike_rate: float
    ball_per_boundary: float
    boundary_percentage: float
    summary: str


def calculate_strike_rate(state: my_state) -> float:
    runs = state["runs"]
    balls = state["balls"]
    strike_rate = (runs / balls) * 100
    return {"strike_rate": strike_rate}


def calculate_ball_per_boundary(state: my_state) -> float:
    balls = state["balls"]
    fours = state["fours"]
    sixes = state["sixes"]
    total_boundaries = fours + sixes
    ball_per_boundary = balls / total_boundaries if total_boundaries > 0 else 0
    return {"ball_per_boundary": ball_per_boundary}


def calculate_boundary_percentage(state: my_state) -> float:
    runs = state["runs"]
    fours = state["fours"]
    sixes = state["sixes"]
    total_boundaries = (fours * 4) + (sixes * 6)
    boundary_percentage = (total_boundaries / runs) * 100 if runs > 0 else 0
    return {"boundary_percentage": boundary_percentage}


def generate_summary(state: my_state) -> str:
    strike_rate = state["strike_rate"]
    ball_per_boundary = state["ball_per_boundary"]
    boundary_percentage = state["boundary_percentage"]
    summary = (
        f"Strike Rate: {strike_rate:.2f}, "
        f"Balls per Boundary: {ball_per_boundary:.2f}, "
        f"Boundary Percentage: {boundary_percentage:.2f}%"
    )
    return {"summary": summary}


graph = StateGraph(my_state)
graph.add_node(calculate_strike_rate)
graph.add_node(calculate_ball_per_boundary)
graph.add_node(calculate_boundary_percentage)
graph.add_node(generate_summary)

graph.add_edge(START, "calculate_strike_rate")
graph.add_edge(START, "calculate_ball_per_boundary")
graph.add_edge(START, "calculate_boundary_percentage")
graph.add_edge("calculate_strike_rate", "generate_summary")
graph.add_edge("calculate_ball_per_boundary", "generate_summary")
graph.add_edge("calculate_boundary_percentage", "generate_summary")
graph.add_edge("generate_summary", END)
result = graph.compile()

initial_state = {
    "runs": 120,
    "balls": 80,
    "fours": 10,
    "sixes": 5,
}
final_state = result.invoke(initial_state)
print("Final State:", final_state)

png = result.get_graph().draw_mermaid_png()

with open("parallel_workflow.png", "wb") as f:
    f.write(png)

print("Workflow saved as parallel_workflow.png")
