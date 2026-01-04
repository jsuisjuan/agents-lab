from src.agents.state import NodeResponse, State
from src.utils.decorators import log_execution
from langchain_core.messages import AIMessage


@log_execution
async def finalize_dialogue(state: State) -> NodeResponse:
    """Sends a final thank you message and sets the 
    finished flag to True.
    """
    name = state.get("name")
    email = state.get("email")
    final_msg = AIMessage(content=f"Thank you, {name}. Your "
        f"registration with email {email} is complete. "
        "Have a great day!")
    return {"messages": [final_msg]}