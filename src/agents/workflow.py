from typing import Annotated, Dict, List, Sequence, Union
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.config import llm


class State(TypedDict):
    """Represents the shared state of the conversation.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    name: str
    email: str
    finished: bool


def greet_user(_state: State) -> Dict[str, List[AIMessage]]:
    """Initiates the converstation by introducing the assistante and
    asking for the user's name.
    """
    msg = AIMessage(content="Hello! I am assistent 360. What is your name?")
    return {"messages": [msg]}


async def collect_name(state: State) -> Dict[str, Union[str, List[AIMessage]]]:
    """Extracts the user's name from the last message and asks for their email.
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage): return {}
    last_user_msg = str(last_msg.content)
    
    response = await llm.ainvoke(f"Extract only the person's name from this text: {last_user_msg}")
    user_name = str(response.content)
    ask_email_msg = AIMessage(content=f"Nice to meet you, {user_name}! Could you please provide your e-mail address?")
    return {"name": user_name, "messages": [ask_email_msg]}


async def collect_email(state: State) -> Dict[str, Union[str, List[AIMessage]]]:
    """Extracts the email from the last message and confirms it to the user.
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage): return {}
    last_user_msg = str(last_msg.content)
    
    response = await llm.ainvoke(f"Extract only the person's email from this text: {last_user_msg}")
    user_email = str(response.content)
    confirmation_msg = AIMessage(content=f"Perfect, I've recorded yout email as: {user_email}.")
    return {"email": user_email, "messages": [confirmation_msg]}


def finalize_dialogue(_state: State) -> Dict[str, Union[str, List[AIMessage]]]:
    """Sends a final thank you message and sets the finished flag to True.
    """
    msg = AIMessage(content="Thank you for the information! Our conversation is now complete.")
    return {"finished": True, "messages": [msg]}


# Graph Construction
memory = MemorySaver()
workflow = StateGraph(State)

# Nodes
workflow.add_node("greet_user", greet_user)
workflow.add_node("collect_name", collect_name)
workflow.add_node("collect_email", collect_email)
workflow.add_node("finalize_dialogue", finalize_dialogue)

# Edges
workflow.add_edge(START, "greet_user")
workflow.add_edge("greet_user", "collect_name")
workflow.add_edge("collect_name", "collect_email")
workflow.add_edge("collect_email", "finalize_dialogue")
workflow.add_edge("finalize_dialogue", END)

# Exportable chain
chain = workflow.compile(
    checkpointer=memory,
    interrupt_after=["greet_user", "collect_name"]
)