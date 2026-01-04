from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.agents.state import State 
from src.agents.edges import route_registration
from src.agents.nodes import (greet_user, finalize_dialogue, 
                              manage_registration)

memory = MemorySaver()
workflow = StateGraph(State)

workflow.add_node("greet_user", greet_user)
workflow.add_node("manage_registration", manage_registration)
workflow.add_node("finalize_dialogue", finalize_dialogue)

workflow.add_edge(START, "greet_user")
workflow.add_edge("greet_user", "manage_registration")
workflow.add_conditional_edges("manage_registration",
    route_registration, {
        "manage_registration": "manage_registration",
        "finalize_dialogue": "finalize_dialogue"})
workflow.add_edge("finalize_dialogue", END)

chain = workflow.compile(checkpointer=memory,
    interrupt_before=["manage_registration"])