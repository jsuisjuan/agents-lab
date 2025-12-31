import sys, asyncio
from src.agents.workflow import chain
from langchain_core.messages import HumanMessage, AIMessage


def gen_workflow_img() -> None:
    """Generates a graph image of the current workflow.
    """
    try:
        graph_png = chain.get_graph().draw_mermaid_png()
        with open("chatbot_gaph.png", "wb") as f:
            f.write(graph_png)
        print("Graph visualization exported to 'chatbot_graph.png'")
    except Exception as e:
        print(f"Could not export graph image:{e}")


async def run_cli() -> None:
    """Run CLI version of the chatbot.
    """
    gen_workflow_img()
    print("\n" + "="*40)
    print("🤖 CHATBOT IA - TERMINAL MODE")
    print("="*40 + "\n")
    
    config = {"configurable": {"thread_id": "local_machine_test"}}
    state = await chain.ainvoke({"messages": []}, config)
    print(f"\nBOT: {state['messages'][-1].content}")
    
    while True:
        user_input = input("\nUSER: ")
        if not user_input.strip(): 
            continue
        
        pre_count = len(state["messages"])
        user_msg = HumanMessage(content=user_input)
        
        chain.update_state(config, {"messages": [user_msg]})
        state = await chain.ainvoke(None, config)
        snapshot_after = chain.get_state(config)
        new_msgs = state["messages"][pre_count + 1:]
        for m in new_msgs:
            if isinstance(m, AIMessage):
                print(f"\nBOT: {m.content}")
        
        if state.get("finished") or snapshot_after.next == ():
            break
    print("\n--- Conversation Finished ---")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\n\nSession ended by user.")
        sys.exit(0)