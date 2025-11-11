import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

load_dotenv()


async def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set")

    # Handle transient errors with exponential backoff
    retry_config = types.HttpRetryOptions(
        attempts=5,  # Maximum retry attempts
        exp_base=7,  # Delay multiplier
        initial_delay=1,  # Initial delay before first retry (in seconds)
        http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
    )

    root_agent = Agent(
        name="helpful_assistant",
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        description="A simple agent that can answer general questions.",
        instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
        tools=[google_search],
    )

    runner = InMemoryRunner(agent=root_agent)

    # response = await runner.run_debug(
    #     "What is Agent Development Kit from Google? What languages is the SDK available in?"
    # )

    response = await runner.run_debug("Who won the last rugby world cup?")

    # print(response)


if __name__ == "__main__":
    asyncio.run(main())
