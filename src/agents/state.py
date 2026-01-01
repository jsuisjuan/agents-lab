from typing import Annotated, List, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages


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