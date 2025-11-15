from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

def get_model(model: str = "gemini-2.5-flash", temperature: float = 0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)