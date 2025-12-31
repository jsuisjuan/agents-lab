import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

def get_model() -> ChatGroq:
    """Return configured LLM instance"""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_retries=3,
        api_key=GROQ_API_KEY
    )

llm = get_model()