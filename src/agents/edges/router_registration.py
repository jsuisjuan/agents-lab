from src.agents.state import State


def route_registration(state: State) -> str:
    """Determines the next workflow node based 
    on the registration status.
    """
    if state.get("info_confirmed"):
        return "finalize_dialogue"
    return "manage_registration"