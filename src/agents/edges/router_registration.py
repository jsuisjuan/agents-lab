from src.agents.state import State


def route_registration(state: State) -> str:
    if state.get("info_confirmed"):
        return "finalize_dialogue"
    return "manage_registration"