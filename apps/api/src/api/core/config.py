from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    GROQ_API_KEY: str
    GEMINI_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    LANGSMITH_API_KEY: str
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

config = Config()