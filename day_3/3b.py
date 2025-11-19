import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.genai import types

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

APP_NAME = "MemoryDemoApp"
USER_ID = "demo_user"

MODEL_NAME = "gemini-2.5-flash-lite"

async def run_session(
    runner_instance: Runner, session_service: InMemorySessionService, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    session: Session | None = None
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content  # type: ignore
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")

async def auto_save_to_memory(callback_context):
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

async def run_memory_agent():
    memory_service = (
        InMemoryMemoryService()
    )  # ADK's built-in Memory Service for development and testing

    # Create agent
    user_agent = LlmAgent(
        model=Gemini(model=MODEL_NAME, retry_options=retry_config),
        name="MemoryDemoAgent",
        instruction="Answer user questions in simple words. Use load_memory tool if you need to recall past conversations.",
        tools=[
            preload_memory
        ],
        after_agent_callback=auto_save_to_memory,  # Saves after each turn!
    )

    # Create Session Service
    session_service = InMemorySessionService()  # Handles conversations

    # Create runner with BOTH services
    runner = Runner(
        agent=user_agent,
        app_name="MemoryDemoApp",
        session_service=session_service,
        memory_service=memory_service,  # Memory service is now available!
    )

    # User tells agent about their favorite color
    # await run_session(
    #     runner,
    #     session_service,
    #     "My favorite color is blue-green. Can you write a Haiku about it?",
    #     "conversation-01",  # Session ID
    # )

    # await run_session(
    #     runner,
    #     session_service,
    #     "What's my favourite color?",
    #     "conversation-02",  # Different session ID
    # )

    await run_session(
        runner,
        session_service,
        "I gifted a new toy to my nephew on his 1st birthday!",
        "auto-save-test",
    )

    # Test 2: Ask about the gift in a NEW session (second conversation)
    # The agent should retrieve the memory using preload_memory and answer correctly
    await run_session(
        runner,
        session_service,
        "What did I gift my nephew?",
        "auto-save-test-2",  # Different session ID - proves memory works across sessions!
    )


async def main(retry_config: types.HttpRetryOptions):
    await run_memory_agent()

if __name__ == "__main__":
    # Handle transient errors with exponential backoff
    retry_config = types.HttpRetryOptions(
        attempts=5,  # Maximum retry attempts
        exp_base=7,  # Delay multiplier
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
    )

    asyncio.run(main(retry_config))