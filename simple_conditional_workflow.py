# conditional_workflow for root calculation of quadratic equation

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class my_state(TypedDict):
    a: int
    b: int
    c: int

    equation: str
    discriminant: float
    result: str


def show_equation(state: my_state):
    result = state['equation'] = f"{state['a']}x^2 + {state['b']}x + {state['c']} = 0"
    return {'equation': result}


def calculate_discriminant(state: my_state):
    discriminant = state['discriminant'] = (
        state['b'] ** 2) - (4 * state['a'] * state['c'])
    return {'discriminant': discriminant}


def real_roots(state: my_state):
    d = state['discriminant']
    a = state['a']
    b = state['b']
    root1 = (-b + (d ** 0.5)) / (2 * a)
    root2 = (-b - (d ** 0.5)) / (2 * a)
    result = state['result'] = f"Roots are real and different: {root1} and {root2}"
    return {'result': result}


def repeated_root(state: my_state):
    root = -state['b']/(2*state['a'])
    result = state['result'] = f"Roots are real and same: {root}"
    return {'result': result}


def complex_roots(state: my_state):
    a = state['a']
    b = state['b']
    c = state['c']
    real_part = -b/(2*a)
    imaginary_part = (abs(state['discriminant'])**0.5)/(2*a)
    result = state['result'] = f"Roots are complex: {real_part} + {imaginary_part}i and {real_part} - {imaginary_part}i"
    return {'result': result}


def check_discriminant(state: my_state):
    d = state['discriminant']
    if d > 0:
        return 'real_roots'
    elif d == 0:
        return 'repeated_root'
    else:
        return 'complex_roots'


graph = StateGraph(my_state)
graph.add_node('show_equation', show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('repeated_root', repeated_root)
graph.add_node('complex_roots', complex_roots)

graph.add_edge(START, 'show_equation')
graph.add_edge('show_equation', 'calculate_discriminant')

graph.add_conditional_edges(
    "calculate_discriminant",
    check_discriminant,
    {
        "real_roots": "real_roots",
        "repeated_root": "repeated_root",
        "complex_roots": "complex_roots",
    },
)


graph.add_edge('real_roots', END)
graph.add_edge('repeated_root', END)
graph.add_edge('complex_roots', END)

workflow = graph.compile()
initial_state = {
    "a": 5,
    "b": -3,
    "c": 8,
}
result = workflow.invoke(initial_state)
print(result)
png = workflow.get_graph().draw_mermaid_png()

with open("conditional_workflow.png", "wb") as f:
    f.write(png)

print("Workflow saved as conditional_workflow.png")
