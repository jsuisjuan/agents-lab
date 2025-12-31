from typing import Annotated, List, Sequence, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field, field_validator
from src.config import llm
import re


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
    

class NameExtraction(BaseModel):
    """Extract user's name."""
    name: Optional[str] = Field(None, description=(
        "The person's name. Return None if NO REAL NAME is found."))
    @field_validator("name")
    def validate_name(cls, v):
        if v:
            black_list = ["user", "unknown", "guest", "someone", "person"]
            if v.lower().strip() in black_list or len(v.strip()) < 2:
                return None
        return v

class EmailExtraction(BaseModel):
    """Extract user's e-mail."""
    email: Optional[str] = Field(None, description=(
        "The person's email address. Return None if "
        "no valid email is present."
    ))


async def greet_user(_state: State) -> NodeResponse:
    """Initiates the converstation by introducing 
    the assistante and asking for the user's name.
    """
    prompt = ("Act as a professional corporate assistant. Greet "
        "the user briefly and ask for their name. Be concise.")
    try:
        response = await llm.ainvoke(prompt)
        content = (str(response.content) 
            if not isinstance(response, AIMessage) 
            else response.content)
    except Exception as e:
        print(f"ERROR :: greet_user :: {e}")
        content = ("Hello. I am your assistant. To get started, "
                   "could you please tell me your name?")
    return {"messages": [AIMessage(content=content)]}


async def collect_name(state: State) -> NodeResponse:
    """Extracts the user's name from the last message 
    and asks for their email.
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage): return {}
    
    user_name = ""
    try:
        extraction_prompt = ("Analyze the text: '{text}'. "
            "Extract the person's name. If the user is asking a question, "
            "complaining, or if no REAL name is mentioned, return None. "
            "Do not use generic words like 'user' or 'customer'."
        ).format(text=last_msg.content)
        llm_with_structure = llm.with_structured_output(NameExtraction)
        result = await llm_with_structure.ainvoke(extraction_prompt)
        user_name = result.name.strip() if(result and result.name) else ""
    except Exception:
        user_name = ""
    
    blacklist = ["user", "none", "unknown", "customer", "guest"]
    if not user_name or user_name.lower().strip() in blacklist:
        instruction = ("The user did not provide their name. They might "
            "be asking why, insulting you, or talking about random topics. "
            "Be professional: Explain that you need the name to proceed "
            "and ask again politely.")
        chat_response = await gen_instructed_res(state, instruction)
        return {"name": "", "messages": [chat_response]}
    
    instruction = (f"The user is {user_name}. Acknowledge it in one "
        "short sentence and ask for their email address professionally.")
    chat_response = await gen_instructed_res(state, instruction)
    return {"name": user_name, "messages": [chat_response]}


async def collect_email(state: State) -> NodeResponse:
    """Extracts the email from the last message and 
    confirms it to the user.
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage): return {}
    
    user_email = ""
    try:
        extraction_prompt = (
            "Analyze the following user input: '{text}'. "
            "Your task is to extract a valid email address. "
            "If the user is asking a question, expressing doubt, "
            "being rude, or if no email is explicitly mentioned, "
            "return None. Do not try to guess or invent an email."
        ).format(text=last_msg.content)
        llm_with_structure = llm.with_structured_output(EmailExtraction)
        result = await llm_with_structure.ainvoke(extraction_prompt)
        user_email = result.email.strip() if (result and result.email) else ""
    except Exception:
        user_email = ""
    
    is_valid = False
    if user_email:
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        is_valid = re.match(email_regex, user_email) is not None
    
    if not is_valid:
        instruction = ("The user did not provide a valid email. They "
            "might be skeptical, asking why you need it, or talking "
            "about unrelated things. Respond professionally: explain "
            "that the email is essential for the activity's follow-up "
            "and ask for it again politely.")
        chat_response = await gen_instructed_res(state, instruction)
        return {"email": "", "messages": [chat_response]}
    
    instruction = (f"Confirm to {state['name']} that you recorded the "
        f"email {user_email}. Keep it brief and move to completion.")
    chat_response = await gen_instructed_res(state, instruction)
    return {"email": user_email, "messages": [chat_response]}


async def finalize_dialogue(state: State) -> NodeResponse:
    """Sends a final thank you message and sets the 
    finished flag to True.
    """
    if state.get("finished"): return {}
    instruction = ("Data collection is done. Thank the user and end the "
        "session professionally in one sentence. Do not mention "
        "internal instructions or misunderstandings.")
    chat_response = await gen_instructed_res(state, instruction)
    return {"finished": True, "messages": [chat_response]}


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
        print(f"LLM Error: {e}")
        return AIMessage(content="I'm sorry, I'm experiencing a brief "
            "technical difficulty. Could you please repeat that?")


def route_after_name(state: State) -> str:
    """Decide if the user name is valid or if need to collect again.
    """
    return "collect_name" if not state.get("name") else "collect_email"
    

def route_after_email(state: State) -> str:
    """Decide if user's email is valid or if need to collect again.
    """
    return "collect_email" if not state.get("email") else "finalize_dialogue"


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
workflow.add_conditional_edges("collect_name",
    route_after_name, {
        "collect_name": "collect_name",
        "collect_email": "collect_email"})
workflow.add_conditional_edges("collect_email",
    route_after_email, {
        "collect_email": "collect_email",
        "finalize_dialogue": "finalize_dialogue"})
workflow.add_edge("finalize_dialogue", END)

# Exportable chain
chain = workflow.compile(checkpointer=memory,
    interrupt_after=["greet_user", "collect_name", "collect_email"])