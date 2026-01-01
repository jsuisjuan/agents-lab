from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.agents.state import State 
from src.agents.edges import route_after_email, route_after_name
from src.agents.nodes import (collect_email, collect_name, 
    greet_user, finalize_dialogue)

memory = MemorySaver()
workflow = StateGraph(State)

workflow.add_node("greet_user", greet_user)
workflow.add_node("collect_name", collect_name)
workflow.add_node("collect_email", collect_email)
workflow.add_node("finalize_dialogue", finalize_dialogue)

workflow.add_edge(START, "greet_user")
workflow.add_edge("greet_user", "collect_name")
workflow.add_conditional_edges("collect_name",
    route_after_name, {
        "collect_name": "collect_name",
        "collect_email": "collect_email"})
workflow.add_conditional_edges("collect_email",
    route_after_email, {
        "collect_email": "collect_email",
        "finalize_dialogue": "finalize_dialogue"})
workflow.add_edge("finalize_dialogue", END)

chain = workflow.compile(checkpointer=memory,
    interrupt_after=["greet_user", "collect_name", "collect_email"])