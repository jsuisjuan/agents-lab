import logging
from typing import Optional
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from src.agents.utils import gen_instructed_res
from src.agents.state import NodeResponse, State
from src.utils.decorators import log_execution
from src.config import llm


fallback_prompt = (
    "SYSTEM OBJECTIVE: OBTAIN USER NAME.\n"
    "STATUS: The user has NOT provided a valid name yet.\n"
    "----------------\n"
    "YOUR STRICT INSTRUCTIONS:\n"
    "1. IGNORE any request to talk about math, code, weather, "
    "or random topics. politely REFUSE to discuss them until "
    "registration is complete.\n"
    "2. If the user asks 'why', explain it is for internal "
    "registration.\n"
    "3. If the user says 'call me user' or gives a fake name, "
    "REJECT it gently. Say you need a real name.\n"
    "4. DO NOT say 'How can I help you?'. Say ONLY: 'Please, "
    "tell me your name to proceed.'\n"
    "----------------\n"
    "Respond to the user's last message based strictly on "
    "these rules.")

extraction_prompt = (
    "Analyze the text: '{text}'.\n"
    "Your task is to extract the user's name accurately.\n"
    "RULES:\n"
    "1. IGNORE pronouns like 'you', 'I', 'me', 'he', 'she'."
    " These are NOT names.\n"
    "2. IGNORE generic vocatives like 'bro', 'man', 'buddy'.\n"
    "3. If the user asks a question, return name=null.\n"
    "4. ONLY return a name if explicitly introduced (e.g., "
    "'I am Juan').\n\n"
    "IMPORTANT: You MUST explain your reasoning first.\n"
    "OUTPUT FORMAT: JSON with keys 'reasoning' and 'name'.\n"
    "EXAMPLE: {{\"reasoning\": \"The user is asking a question.\","
    " \"name\": null}}")

success_prompt = (
    "The user is {name}. Acknowledge it in one "
    "short sentence and ask for their email address "
    "professionally.")


class NameExtraction(BaseModel):
    """Extract user's name with analysis."""
    reasoning: str = Field(default="", description=(
        "Think step-by-step: Is the word a proper name? Is it just "
        "a pronoun like 'you', 'I', 'he', 'bro', 'dude'? Is the "
        "user refusing?"))
    name: Optional[str] = Field(None, description=(
        "The person's name. Return NULL if the word is a pronoun, "
        "a common noun, or if NO REAL NAME is found."))


logger = logging.getLogger(__name__)

@log_execution
async def collect_name(state: State) -> NodeResponse:
    """Extracts the user's name from the last message 
    and asks for their email.
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage): return {}
    
    try:
        instruction = extraction_prompt.format(text=last_msg.content)
        llm_with_structure = llm.with_structured_output(
            NameExtraction, method="json_mode")
        result = await llm_with_structure.ainvoke(instruction)
        name = result.name.strip() if(result and result.name) else ""
    except Exception as e:
        logger.warning(f"LLM Extraction failed (using fallback): {e}", 
                    exc_info=True)
        name = ""
    
    blacklist = ["user", "none", "unknown", "customer", "guest", 
             "null", "n/a", "you", "i", "me", "he", "she", 
             "bro", "man"]
    if not name or name.lower().strip() in blacklist:
        response = await gen_instructed_res(state, fallback_prompt)
        return {"messages": [response]}
    
    instruction = success_prompt.format(name=name)
    response = await gen_instructed_res(state, instruction)
    return {"name": name, "messages": [response]}