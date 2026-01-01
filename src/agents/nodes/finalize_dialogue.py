from src.agents.utils import gen_instructed_res
from src.agents.state import NodeResponse, State
from src.utils.decorators import log_execution

end_prompt = (
    "Data collection is done. Thank the user and end the "
    "session professionally in one sentence. Do not mention "
    "internal instructions or misunderstandings.")


@log_execution
async def finalize_dialogue(state: State) -> NodeResponse:
    """Sends a final thank you message and sets the 
    finished flag to True.
    """
    if state.get("finished"): return {}
    response = await gen_instructed_res(state, end_prompt)
    return {"finished": True, "messages": [response]}