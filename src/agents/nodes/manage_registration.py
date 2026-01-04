import logging, re
from typing import Optional, Tuple
from pydantic import BaseModel, Field
from src.config import llm
from src.utils.decorators import log_execution
from src.agents.state import State, NodeResponse
from src.agents.utils import gen_instructed_res
from langchain_core.messages import AIMessage, BaseMessage


logger = logging.getLogger(__name__)
ASK_NAME_INSTRUCTION = (
    "STATUS: User Name is MISSING.\n"
    "GOAL: Ask for the user's name.\n"
    "CONSTRAINT: Ignore any other topic or question the user asked. "
    "If they ask 'why', say it's for registration. Do not answer "
    "random questions."
)

ASK_EMAIL_INSTRUCTION = (
    "STATUS: Name is '{name}'. Email is MISSING.\n"
    "GOAL: Ask for the email address.\n"
    "CONSTRAINT: Ignore user chatter. Focus ONLY on getting the email."
)

CONFIRM_DATA_INSTRUCTION = (
    "STATUS: Data collected (Name={name}, Email={email}).\n"
    "GOAL: Say: '{prefix}I have Name: {name} and Email: {email}. "
    "Is this correct?'\n"
    "CONSTRAINT: Do not talk about anything else. Force a "
    "Yes/No confirmation."
)

EXTRACTION_PROMPT = (
    "You are a Data Extraction Engine. NOT a chatbot.\n"
    "Current Data -> Name: {name} | Email: {email}\n"
    "TASK: Extract Name and Email updates from user input.\n"
    "RULES:\n"
    "1. If user gives/corrects Name, extract to 'name'.\n"
    "2. If user gives/corrects Email, extract to 'email'.\n"
    "3. Ignore confirmation words like 'ok', 'yes' here.\n"
    "IMPORTANT: Respond in JSON."
)

CONFIRMATION_PROMPT = (
    "You are a Confirmation Analyzer.\n"
    "Context: The bot asked 'Is the data correct?'\n"
    "User Input: '{user_input}'\n"
    "TASK: Determine if the user is confirming/agreeing or exiting.\n"
    "RULES:\n"
    "1. Return is_confirmed=True for: 'yes', 'ok', 'correct', 'right'"
    ", 'bye', 'thanks', 'sure', 'yea.\n"
    "2. Return is_confirmed=False for: 'no', 'wrong', 'change name', "
    "'change email' or random chatter.\n"
    "IMPORTANT: Respond in JSON."
)


class DataUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Extracted name.")
    email: Optional[str] = Field(None, description="Extracted email.")

class ConfirmationCheck(BaseModel):
    is_confirmed: bool = Field(False, description="True if user confirms.")


async def extract_data(
    name: Optional[str], 
    email: Optional[str], 
    last_msg: BaseMessage
) -> Tuple[Optional[str], Optional[str]]:
    """Extracts name and email updates from the user's 
    message using an LLM.
    Args:
        name: The current name stored in state.
        email: The current email stored in state.
        last_msg: The most recent message from the user.
    Returns:
        A tuple containing (new_name, new_email).
    """
    try:
        ex_prompt = EXTRACTION_PROMPT.format(name=name, email=email)
        full_ex_prompt = f"{ex_prompt}\n\nUser Input: '{last_msg.content}'"
        
        llm_extract = llm.with_structured_output(DataUpdate, 
                                                 method="json_mode")
        result_data = await llm_extract.ainvoke(full_ex_prompt)
        
        new_name = result_data.name if result_data.name else name
        new_email = email
        if result_data.email:
            if re.search(r'[^@]+@[^@]+\.[^@]+', result_data.email):
                new_email = result_data.email
    except Exception as e:
        logger.error(f"Data Extraction Failed: {e}")
        new_name, new_email = name, email
    return new_name, new_email


async def verify_confirmation(
    new_name: Optional[str], 
    name: Optional[str], 
    new_email: Optional[str], 
    email: Optional[str], 
    last_msg: BaseMessage
) -> Tuple[bool, bool]:
    """Checks if the user confirmed the data using Regex or LLM analysis.
    It enforces a rule where confirmation is invalid if data changed 
    in the current turn.
    Returns:
        A tuple containing (is_confirmed, data_changed).
    """
    data_changed = (new_name != name) or (new_email != email)
    is_confirmed = False
    if new_name and new_email and not data_changed:
        regex_yes = re.search(r"\b(yes|yep|ok|okay|correct|right|"
            "confirm|bye|sim|tá)\b", last_msg.content, re.IGNORECASE)
        if regex_yes:
            logger.info(f"Regex Confirmation Detected: '{regex_yes.group(0)}'")
            is_confirmed = True
        else:
            try:
                conf_prompt_fmt = CONFIRMATION_PROMPT.format(
                    user_input=last_msg.content)
                llm_confirm = llm.with_structured_output(ConfirmationCheck, 
                                                         method="json_mode")
                result_conf = await llm_confirm.ainvoke(conf_prompt_fmt)
                is_confirmed = result_conf.is_confirmed
            except Exception as e:
                logger.warning(f"Confirmation Check Failed: {e}")
                is_confirmed = False
    elif data_changed:
        logger.info("Data changed in this turn. Forcing summary review.")
        is_confirmed = False
    return is_confirmed, data_changed


@log_execution
async def manage_registration(state: State) -> NodeResponse:
    """Main node logic to orchestrate data collection 
    and validation loop.
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage): return {}
    name, email = state.get("name"), state.get("email")
    new_name, new_email = await extract_data(name, email, last_msg)
    is_confirmed, data_changed = await verify_confirmation(
        new_name, name, new_email, email, last_msg)
    
    if is_confirmed and new_name and new_email:
        return {"name": new_name, "email": new_email, 
                "info_confirmed": True}
    
    if not new_name:
        instruction = ASK_NAME_INSTRUCTION
    elif not new_email:
        instruction = ASK_EMAIL_INSTRUCTION.format(name=new_name)
    else:
        prefix = "I've updated your info. " if data_changed else ""
        instruction = CONFIRM_DATA_INSTRUCTION.format(
            name=new_name, email=new_email, prefix=prefix)

    response = await gen_instructed_res(state, instruction)
    return {"name": new_name, "email": new_email, 
            "info_confirmed": False, "messages": [response]}