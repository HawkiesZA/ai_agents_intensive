
import asyncio
import os
from dotenv import load_dotenv

import sqlite3

from typing import Any, Dict

from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

APP_NAME = "default" 
USER_ID = "default"
SESSION = "default"

MODEL_NAME = "gemini-2.5-flash-lite"

# Define scope levels for state keys (following best practices)
USER_NAME_SCOPE_LEVELS = ("temp", "user", "app")


# This demonstrates how tools can write to session state using tool_context.
# The 'user:' prefix indicates this is user-specific data.
def save_userinfo(
    tool_context: ToolContext, user_name: str, country: str
) -> Dict[str, Any]:
    """
    Tool to record and save user name and country in session state.

    Args:
        user_name: The username to store in session state
        country: The name of the user's country
    """
    # Write to session state using the 'user:' prefix for user data
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country

    return {"status": "success"}


# This demonstrates how tools can read from session state.
def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve user name and country from session state.
    """
    # Read from session state
    user_name = tool_context.state.get("user:name", "Username not found")
    country = tool_context.state.get("user:country", "Country not found")

    return {"status": "success", "user_name": user_name, "country": country}

# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    session_service: InMemorySessionService,
    user_queries: list[str] | str | None = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    session: Session | None = None
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            queryContent = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=queryContent  # type: ignore
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")

async def run_forgetful_ai():
    # Step 1: Create the LLM Agent
    root_agent = Agent(
        model=Gemini(model=MODEL_NAME, retry_options=retry_config),
        name="text_chat_bot",
        description="A text chatbot",  # Description of the agent's purpose
    )

    # Step 2: Set up Session Management
    # InMemorySessionService stores conversations in RAM (temporary)
    session_service = InMemorySessionService()

    # Step 3: Create the Runner
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    # Run a conversation with two queries in the same session
    # Notice: Both queries are part of the SAME session, so context is maintained
    await run_session(
        runner,
        session_service,
        [
            "Hi, I am Sam! What is the capital of United States?",
            "Hello! What is my name?",  # This time, the agent should remember!
        ],
        "stateful-agentic-session",
    )

async def run_persistent_ai():
    # Step 1: Create the same agent (notice we use LlmAgent this time)
    chatbot_agent = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="text_chat_bot",
        description="A text chatbot with persistent memory",
    )

    # Step 2: Switch to DatabaseSessionService
    # SQLite database will be created automatically
    db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
    session_service = DatabaseSessionService(db_url=db_url)

    # Step 3: Create a new runner with persistent storage
    runner = Runner(agent=chatbot_agent, app_name=APP_NAME, session_service=session_service)

    # Stores session in sqlite db
    await run_session(
        runner,
        session_service,
        ["Hi, I am Sam! What is the capital of the United States?", "Hello! What is my name?"],
        "test-db-session-01",
    )

    # Retrieves session from sqlite db
    # await run_session(
    #     runner,
    #     session_service,
    #     ["What is the capital of India?", "Hello! What is my name?"],
    #     "test-db-session-01",
    # )

    # Can't retrieve previous session because of a new session name - sessions are isolated
    # await run_session(
    #     runner,
    #     session_service,
    #     ["Hello! What is my name?"],
    #     "test-db-session-02"
    # )  # Note, we are using new session name

async def run_persistent_ai_with_compacting():
    chatbot_agent = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="text_chat_bot",
        description="A text chatbot with persistent memory",
    )
    
    # Re-define our app with Events Compaction enabled
    research_app_compacting = App(
        name="research_app_compacting",
        root_agent=chatbot_agent,
        # This is the new part!
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=3,  # Trigger compaction every 3 invocations
            overlap_size=1,  # Keep 1 previous turn for context
        ),
    )

    db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
    session_service = DatabaseSessionService(db_url=db_url)

    # Create a new runner for our upgraded app
    research_runner_compacting = Runner(
        app=research_app_compacting, session_service=session_service
    )

    # Turn 1
    await run_session(
        research_runner_compacting,
        session_service,
        "What is the latest news about AI in healthcare?",
        "compaction_demo",
    )

    # Turn 2
    await run_session(
        research_runner_compacting,
        session_service,
        "Are there any new developments in drug discovery?",
        "compaction_demo",
    )

    # Turn 3 - Compaction should trigger after this turn!
    await run_session(
        research_runner_compacting,
        session_service,
        "Tell me more about the second development you found.",
        "compaction_demo",
    )

    # Turn 4
    await run_session(
        research_runner_compacting,
        session_service,
        "Who are the main companies involved in that?",
        "compaction_demo",
    )

    await get_final_session_state(research_runner_compacting, session_service)

async def run_persistent_ai_with_compacting_and_tools():
    # Create an agent with session state tools
    root_agent = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="text_chat_bot",
        description="""A text chatbot.
        Tools for managing user context:
        * To record username and country when provided use `save_userinfo` tool. 
        * To fetch username and country when required use `retrieve_userinfo` tool.
        """,
        tools=[save_userinfo, retrieve_userinfo],  # Provide the tools to the agent
    )

    # Set up session service and runner
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service, app_name="default")

    # Test conversation demonstrating session state
    await run_session(
        runner,
        session_service,
        [
            "Hi there, how are you doing today? What is my name?",  # Agent shouldn't know the name yet
            "My name is Sam. I'm from Poland.",  # Provide name - agent should save it
            "What is my name? Which country am I from?",  # Agent should recall from session state
        ],
        "state-demo-session",
    )
    await print_session(session_service)

    # Start a completely new session - the agent won't know our name
    await run_session(
        runner,
        session_service,
        ["Hi there, how are you doing today? What is my name?"],
        "new-isolated-session",
    )

    await print_session(session_service, "new-isolated-session")

async def print_session(session_service: InMemorySessionService, session_id: str = "state-demo-session"):
    # Retrieve the session and inspect its state
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )

    print(f"Session State Contents for {session_id}:")
    print(session.state) # type: ignore
    print("\n🔍 Notice the 'user:name' and 'user:country' keys storing our data!")

def check_data_in_db():
    with sqlite3.connect("my_agent_data.db") as connection:
        cursor = connection.cursor()
        result = cursor.execute(
            "select app_name, session_id, author, content from events"
        )
        print([_[0] for _ in result.description])
        for each in result.fetchall():
            print(each)

async def get_final_session_state(
    research_runner_compacting: Runner,
    session_service: DatabaseSessionService
    ):
    """
    This function searches for the compaction summary event in the final session.
    """
    final_session = await session_service.get_session(
        app_name=research_runner_compacting.app_name,
        user_id=USER_ID,
        session_id="compaction_demo",
    )

    print("--- Searching for Compaction Summary Event ---")
    found_summary = False
    for event in final_session.events: # type: ignore
        # Compaction events have a 'compaction' attribute
        if event.actions and event.actions.compaction:
            print("\n✅ SUCCESS! Found the Compaction Event:")
            print(f"  Author: {event.author}")
            print(f"\n Compacted information: {event}")
            found_summary = True
            break

    if not found_summary:
        print(
            "\n❌ No compaction event found. Try increasing the number of turns in the demo."
        )

def cleanup():
    if os.path.exists("my_agent_data.db"):
        os.remove("my_agent_data.db")

async def main(retry_config: types.HttpRetryOptions):
    #await run_forgetful_ai()
    #await run_persistent_ai()

    #await run_persistent_ai_with_compacting()
    #check_data_in_db()

    await run_persistent_ai_with_compacting_and_tools()

if __name__ == "__main__":
    # Handle transient errors with exponential backoff
    retry_config = types.HttpRetryOptions(
        attempts=5,  # Maximum retry attempts
        exp_base=7,  # Delay multiplier
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
    )

    asyncio.run(main(retry_config))