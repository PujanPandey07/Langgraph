from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import operator

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)


class poststate(TypedDict):
    topic: str
    tweet: str
    feedback: str
    iteration: int
    max_iterations: int
    evaluation: Literal["approved", "needs improvement"]

    tweet_histroy: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]


class outputschema(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        ..., description="Final evaluation result.")
    feedback: str = Field(..., description="feedback for the tweet.")


structured_llm = llm.with_structured_output(outputschema)


def generate_tweet(state: poststate):

    # prompt
    messages = [
        SystemMessage(
            content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ]

    # send generator_llm
    response = llm.invoke(messages).content

    # return response
    return {'tweet': response, 'tweet_history': [response]}


def evaluate_tweet(state: poststate):
    # prompt
    messages = [
        SystemMessage(
            content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
        HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality – Is this fresh, or have you seen it a hundred times before?  
2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
3. Punchiness – Is it short, sharp, and scroll-stopping?  
4. Virality Potential – Would people retweet or share it?  
5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"  
- feedback: One paragraph explaining the strengths and weaknesses 
""")
    ]

    response = structured_llm.invoke(messages)

    return {'evaluation': response.evaluation, 'feedback': response.feedback, 'feedback_history': [response.feedback]}


def optimize_tweet(state: poststate):
    messages = [
        SystemMessage(
            content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]

    response = llm.invoke(messages).content
    iteration = state['iteration'] + 1

    return {'tweet': response, 'iteration': iteration, 'tweet_history': [response]}


def route_evaluation(state: poststate):
    if state['evaluation'] == 'approved':
        return 'approved'
    else:
        if state['iteration'] >= state['max_iterations']:
            return 'approved'
        else:
            return 'needs improvement'


initial_state = {
    "topic": "football",
    "iteration": 1,
    "max_iteration": 5
}


graph = StateGraph(poststate)
graph.add_node('generate_tweet', generate_tweet)
graph.add_node('evaluate_tweet', evaluate_tweet)
graph.add_node('optimize_tweet', optimize_tweet)

graph.add_edge(START, 'generate_tweet')
graph.add_edge('generate_tweet', 'evaluate_tweet')
graph.add_conditional_edges('evaluate_tweet', route_evaluation, {
    'approved': END,
    'needs improvement': 'optimize_tweet'
})

workflow = graph.compile()
result = workflow.invoke(initial_state)
print("Final State:", result)
png = workflow.get_graph().draw_mermaid_png()

with open("iterative_workflow.png", "wb") as f:
    f.write(png)

print("Workflow saved as iterative_workflow.png")
