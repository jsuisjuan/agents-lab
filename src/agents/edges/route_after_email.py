
from src.agents.workflow import State


def route_after_email(state: State) -> str:
    """Decide if user's email is valid or if need to collect again.
    """
    email = state.get("email")
    hasnt_email = not email or email.strip() == ""
    return "collect_email" if hasnt_email else "finalize_dialogue"