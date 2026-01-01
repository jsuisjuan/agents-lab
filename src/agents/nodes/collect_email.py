from langchain_core.messages import AIMessage
from src.utils.decorators import log_execution
from src.agents.utils import gen_instructed_res
from src.agents.state import NodeResponse, State
import re


fallback_prompt = (
    "The user did not provide a valid email. They "
    "might be skeptical, asking why you need it, or talking "
    "about unrelated things. Respond professionally: explain "
    "that the email is essential for the activity's follow-up "
    "and ask for it again politely.")


success_prompt = (
    "Confirm to {name} that you recorded the "
    "email {email}. Keep it brief and professional.")


@log_execution
async def collect_email(state: State) -> NodeResponse:
    """Extracts the email from the last message and 
    confirms it to the user.
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage): return {}
    
    email_regex = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    matches = re.findall(email_regex, last_msg.content)
    email = matches[0] if matches else ""
    if not email:
        response = await gen_instructed_res(state, fallback_prompt)
        return {"messages": [response]}
    
    instruction = success_prompt.format(name=state["name"], email=email)
    response = await gen_instructed_res(state, instruction)
    return {"email": email, "messages": [response]}