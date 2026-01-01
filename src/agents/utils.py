import logging
from langchain_core.messages import AIMessage, SystemMessage
from src.agents.workflow import State
from src.config import llm


logger = logging.getLogger(__name__)

async def gen_instructed_res(state: State, instruction: str) -> AIMessage:
    """Combines conversation history with a specific system instruction 
    to generate a dynamic AI response.
    """
    try:
        full_instruction = (f"{instruction} Always be concise and "
                            "professional. Avoid small talk.")
        system_msg = SystemMessage(content=full_instruction)
        chat_prompt = [*state["messages"], system_msg]
        response = await llm.ainvoke(chat_prompt)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response.content))
        return response
    except Exception as e:
        logger.error(f"Failed to generate instructed response: {e}", 
                     exc_info=True)
        return AIMessage(content="I'm sorry, I'm experiencing a brief "
                "technical difficulty. Could you please repeat that?")
