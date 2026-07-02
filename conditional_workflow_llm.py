from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from pydantic import BaseModel, Field


# -----------------------
# LLM
# -----------------------
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)


# -----------------------
# Structured Output Schemas
# -----------------------
class ReviewSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the review."
    )


class DiagnosisSchema(BaseModel):
    issue_type: str = Field(description="Main issue mentioned in the review.")
    tone: str = Field(description="Customer's emotional tone.")
    urgency: str = Field(description="Urgency level: Low, Medium, or High.")


sentiment_llm = llm.with_structured_output(ReviewSchema)
diagnosis_llm = llm.with_structured_output(DiagnosisSchema)


# -----------------------
# State
# -----------------------
class MyState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str


# -----------------------
# Nodes
# -----------------------
def find_sentiment(state: MyState):
    prompt = f"""
Analyze the following customer review and determine whether it is
positive or negative.

Review:
{state["review"]}
"""

    result = sentiment_llm.invoke(prompt)

    return {
        "sentiment": result.sentiment
    }


def check_sentiment(state: MyState):
    if state["sentiment"] == "positive":
        return "positive_response"

    return "run_diagnosis"


def positive_response(state: MyState):
    prompt = f"""
Write a warm, friendly thank-you response to this positive review.

Review:
{state["review"]}
"""

    response = llm.invoke(prompt)

    return {
        "response": response.content
    }


def run_diagnosis(state: MyState):
    prompt = f"""
Analyze the following negative customer review.

Return:
- issue_type
- tone
- urgency

Review:
{state["review"]}
"""

    diagnosis = diagnosis_llm.invoke(prompt)

    return {
        "diagnosis": diagnosis.model_dump()
    }


def negative_response(state: MyState):
    diagnosis = state["diagnosis"]

    prompt = f"""
You are a customer support assistant.

The customer experienced a "{diagnosis['issue_type']}" issue.

Their tone was "{diagnosis['tone']}".

Urgency level is "{diagnosis['urgency']}".

Write an empathetic, professional response that:
- acknowledges the problem,
- apologizes,
- offers help,
- reassures the customer.
"""

    response = llm.invoke(prompt)

    return {
        "response": response.content
    }


# -----------------------
# Graph
# -----------------------
graph = StateGraph(MyState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")

graph.add_conditional_edges(
    "find_sentiment",
    check_sentiment,
    {
        "positive_response": "positive_response",
        "run_diagnosis": "run_diagnosis",
    },
)

graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("positive_response", END)
graph.add_edge("negative_response", END)

workflow = graph.compile()


# -----------------------
# Test
# -----------------------
initial_state = {
    "review": "The product quality is terrible and the delivery was late."
}

result = workflow.invoke(initial_state)

print("\nFinal State:")
print(result)


# -----------------------
# Save Graph Image
# -----------------------
png = workflow.get_graph().draw_mermaid_png()

with open("conditional_workflow_llm.png", "wb") as f:
    f.write(png)

print("\nWorkflow saved as 'conditional_workflow_llm.png'")
