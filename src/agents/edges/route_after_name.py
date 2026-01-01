
from src.agents.workflow import State


def route_after_name(state: State) -> str:
    """Decide if the user name is valid or if need to collect again.
    """
    name = state.get("name")
    hasnt_name = not name or name.strip() == ""
    return "collect_name" if hasnt_name else "collect_email"