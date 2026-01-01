import logging
from langchain_core.messages import AIMessage
from src.agents.state import NodeResponse, State
from src.utils.decorators import log_execution
from src.config import llm


greet_prompt = (
    "Act as a professional corporate assistant. Greet "
    "the user briefly and ask for their name. Be concise.")


logger = logging.getLogger(__name__)

@log_execution
async def greet_user(_state: State) -> NodeResponse:
    """Initiates the converstation by introducing 
    the assistante and asking for the user's name.
    """
    try:
        response = await llm.ainvoke(greet_prompt)
        content = (str(response.content) 
            if not isinstance(response, AIMessage) 
            else response.content)
    except Exception as e:
        logger.error(f"Failed to greet user: {e}", exc_info=True)
        content = ("Hello. I am your assistant. To get started, "
                   "could you please tell me your name?")
    return {"messages": [AIMessage(content=content)]}