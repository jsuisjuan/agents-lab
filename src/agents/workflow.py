from typing import Annotated, List, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
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

class NodeResponse(TypedDict, total=False):
    """Represents what the node can return.
    """
    messages: List[AIMessage]
    name: str
    email: str
    finished: bool


async def greet_user(_state: State) -> NodeResponse:
    """Initiates the converstation by introducing 
    the assistante and asking for the user's name.
    """
    prompt = ("You are a friendly assistant. Greet the "
        "user and politely ask for their name.")
    response = await llm.ainvoke(prompt)
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response.content))
    return {"messages": [response]}


async def collect_name(state: State) -> NodeResponse:
    """Extracts the user's name from the last 
    message and asks for their email.
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage): return {}
    
    response = await llm.ainvoke("Extract only the person's name "
        f"from this text: {str(last_msg.content)}")
    user_name = str(response.content)
    
    instruction = (f"The user's name is {user_name}. "
        "Acknowledge it warmly and ask for their email.")
    chat_response = await gen_instructed_res(state, instruction)
    return {"name": user_name, "messages": [chat_response]}


async def collect_email(state: State) -> NodeResponse:
    """Extracts the email from the last message and 
    confirms it to the user.
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage): return {}
    last_user_msg = str(last_msg.content)
    
    response = await llm.ainvoke("Extract only the person's email "
        f"from this text: {last_user_msg}")
    user_email = str(response.content)
    
    instruction = ("Acknowledge it warmly that you "
        f"receive and recorded his email: {user_email}.")
    chat_response = await gen_instructed_res(state, instruction)
    return {"email": user_email, "messages": [chat_response]}


async def finalize_dialogue(state: State) -> NodeResponse:
    """Sends a final thank you message and sets the 
    finished flag to True.
    """
    instruction = ("The data collection is complete. Thank the user politely, "
        "mention the process is finished, and say goodbye warmly.")
    chat_response = await gen_instructed_res(state, instruction)
    return {"finished": True, "messages": [chat_response]}


async def gen_instructed_res(state: State, instruction: str) -> AIMessage:
    """Combines conversation history with a specific system instruction 
    to generate a dynamic AI response.
    """
    system_msg = SystemMessage(content=instruction)
    chat_prompt = [*state["messages"], system_msg]
    response = await llm.ainvoke(chat_prompt)
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response.content))
    return response


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
chain = workflow.compile(checkpointer=memory,
    interrupt_after=["greet_user", "collect_name"])