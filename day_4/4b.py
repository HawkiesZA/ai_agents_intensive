import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

APP_NAME = "default" 
USER_ID = "default"
SESSION = "default"

MODEL_NAME = "gemini-2.5-flash-lite"