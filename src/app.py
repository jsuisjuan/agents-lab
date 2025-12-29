import uvicorn
from typing import Any, Dict
from fastapi import FastAPI, Request, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.workflow import chain


app = FastAPI(title="ChatbotIA")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "online"}


@app.post("/webhook")
async def whatsapp_webhook(request: Request) -> Dict[str, Any]:
    """Endpoint for receiving WhatsApp messages.
    """
    try:
        data = await request.json()
        user_id = data.get("sender") or data.get("from")
        user_text = data.get("text") or data.get("message")
        if not user_id or not user_text:
            raise HTTPException(status_code=400, detail="Invalid payload")
        
        config = {"configurable": {"thread_id": str(user_id)}}
        current_state = await chain.aget_state(config)
        pre_count = (len(current_state.values.get("messages", [])) 
                     if current_state.values else 0)
        
        user_msg = HumanMessage(content=user_text)
        chain.update_state(config, {"messages": [user_msg]})
        
        final_state = await chain.ainvoke(None, config)
        new_response = [m.content for m 
            in final_state["messages"][pre_count + 1:] 
            if isinstance(m, AIMessage)]
        
        return {"user": user_id, "responses": new_response, 
            "finished": final_state.get("finished", False)}
    except Exception as e:
        print(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)