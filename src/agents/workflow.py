import sys
import asyncio
from typing import Dict, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from src.config import llm


class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str


async def generate_joke(state: State) -> Dict[str, str]:
    """First LLM call to generate initial joke"""
    msg = await llm.ainvoke(f"Write a short joke about {state['topic']}")
    return {"joke": str(msg.content)}


def check_punchline(state: State) -> Literal["Pass", "Fail"]:
    """Gate function to check if the joke has a punchline"""
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"


async def improve_joke(state: State) -> Dict[str, str]:
    """Second LLM call to improve the joke"""
    msg = await llm.ainvoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": str(msg.content)}


async def polish_joke(state: State) -> Dict[str, str]:
    """Third LLM call for final polish"""
    msg = await llm.ainvoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": str(msg.content)}


workflow = StateGraph(State)

workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)


async def main() -> None:
    chain = workflow.compile()
    try:
        graph_png = chain.get_graph().draw_mermaid_png()
        with open("chatbot_gaph.png", "wb") as f:
            f.write(graph_png)
        print("'chatbot_graph.png' was exported successfully")
    except Exception as e:
        print(f"Could not export 'chatbot_graph.png': {e}")
    
    state = await chain.ainvoke({"topic": "cats"})
    print(f"Initial joke:\n{state['joke']}\n--- --- ---\n")
    if "improved_joke" in state:
        print(f"Improved joke:\n{state['improved_joke']}\n--- --- ---\n")
        print(f"Final joke:\n{state['final_joke']}")
    else:
        print(f"Final joke:\n{state['joke']}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass