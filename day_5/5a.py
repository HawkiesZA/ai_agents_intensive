import json
import requests
import subprocess
import time
import uuid
import asyncio
import logging
import os
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

APP_NAME = "default" 
USER_ID = "default"
SESSION = "default"

MODEL_NAME = "gemini-2.5-flash-lite"

def start_product_catalog_server():
    server_process = subprocess.Popen(
        [
            "uvicorn",
            "product_catalog_server:app",  # Module:app format
            "--host",
            "localhost",
            "--port",
            "8001",
        ],
        cwd="day_5/product_catalog_server",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ},  # Pass environment variables (including GOOGLE_API_KEY)
    )

    # Wait for server to start (poll until it responds)
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get(
                "http://localhost:8001/.well-known/agent-card.json", timeout=1
            )
            if response.status_code == 200:
                print(f"\n✅ Product Catalog Agent server is running!")
                print(f"   Server URL: http://localhost:8001")
                print(f"   Agent card: http://localhost:8001/.well-known/agent-card.json")
                break
        except requests.exceptions.RequestException:
            time.sleep(5)
            print(".", end="", flush=True)
    else:
        print("\n⚠️  Server may not be ready yet. Check manually if needed.")

def fetch_agent_card():
    try:
        response = requests.get(
            "http://localhost:8001/.well-known/agent-card.json", timeout=5
        )

        if response.status_code == 200:
            agent_card = response.json()
            print("📋 Product Catalog Agent Card:")
            print(json.dumps(agent_card, indent=2))

            print("\n✨ Key Information:")
            print(f"   Name: {agent_card.get('name')}")
            print(f"   Description: {agent_card.get('description')}")
            print(f"   URL: {agent_card.get('url')}")
            print(f"   Skills: {len(agent_card.get('skills', []))} capabilities exposed")
        else:
            print(f"❌ Failed to fetch agent card: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching agent card: {e}")
        print("   Make sure the Product Catalog Agent server is running (previous cell)")

def create_remote_product_catalog_agent() -> RemoteA2aAgent:
    # This acts as a client-side proxy - the Customer Support Agent can use it like a local agent
    remote_product_catalog_agent = RemoteA2aAgent(
        name="product_catalog_agent",
        description="Remote product catalog agent from external vendor that provides product information.",
        # Point to the agent card URL - this is where the A2A protocol metadata lives
        agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
    )

    return remote_product_catalog_agent

def create_customer_support_agent(remote_product_catalog_agent: RemoteA2aAgent, retry_config: types.HttpRetryOptions) -> LlmAgent:
    customer_support_agent = LlmAgent(
        model=Gemini(model=MODEL_NAME, retry_options=retry_config),
        name="customer_support_agent",
        description="A customer support assistant that helps customers with product inquiries and information.",
        instruction="""
        You are a friendly and professional customer support agent.
        
        When customers ask about products:
        1. Use the product_catalog_agent sub-agent to look up product information
        2. Provide clear answers about pricing, availability, and specifications
        3. If a product is out of stock, mention the expected availability
        4. Be helpful and professional!
        
        Always get product information from the product_catalog_agent before answering customer questions.
        """,
        sub_agents=[remote_product_catalog_agent],  # Add the remote agent as a sub-agent!
    )
    return customer_support_agent

async def test_a2a_communication(user_query: str, customer_support_agent: LlmAgent):
    """
    Test the A2A communication between Customer Support Agent and Product Catalog Agent.

    This function:
    1. Creates a new session for this conversation
    2. Sends the query to the Customer Support Agent
    3. Support Agent communicates with Product Catalog Agent via A2A
    4. Displays the response

    Args:
        user_query: The question to ask the Customer Support Agent
    """
    # Setup session management (required by ADK)
    session_service = InMemorySessionService()

    # Session identifiers
    app_name = "agent"
    user_id = "demo_user"
    # Use unique session ID for each test to avoid conflicts
    session_id = f"demo_session_{uuid.uuid4().hex[:8]}"

    # CRITICAL: Create session BEFORE running agent (synchronous, not async!)
    # This pattern matches the deployment notebook exactly
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    # Create runner for the Customer Support Agent
    # The runner manages the agent execution and session state
    runner = Runner(
        agent=customer_support_agent, app_name=app_name, session_service=session_service
    )

    # Create the user message
    # This follows the same pattern as the deployment notebook
    test_content = types.Content(parts=[types.Part(text=user_query)])

    # Display query
    print(f"\n👤 Customer: {user_query}")
    print(f"\n🎧 Support Agent response:")
    print("-" * 60)

    # Run the agent asynchronously (handles streaming responses and A2A communication)
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=test_content
    ):
        # Print final response only (skip intermediate events)
        if event.is_final_response() and event.content:
            for part in event.content.parts:  # type: ignore
                if hasattr(part, "text"):
                    print(part.text)

    print("-" * 60)

async def main(retry_config: types.HttpRetryOptions):
    start_product_catalog_server()
    fetch_agent_card()

    remote_product_catalog_agent = create_remote_product_catalog_agent()
    customer_support_agent = create_customer_support_agent(remote_product_catalog_agent, retry_config)

    await test_a2a_communication("Can you tell me about the iPhone 15 Pro? Is it in stock?", customer_support_agent)

    # Test comparing multiple products
    await test_a2a_communication(
        "I'm looking for a laptop. Can you compare the Dell XPS 15 and MacBook Pro 14 for me?",
        customer_support_agent
    )

    # Test specific product inquiry
    await test_a2a_communication(
        "Do you have the Sony WH-1000XM5 headphones? What's the price?",
        customer_support_agent
    )

    # Test out of stock product
    await test_a2a_communication(
        "I need an ultrawide monitor. Can you tell me about it?",
        customer_support_agent
    )


if __name__ == "__main__":
    retry_config = types.HttpRetryOptions(
        attempts=5,  # Maximum retry attempts
        exp_base=7,  # Delay multiplier
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
    )
    asyncio.run(main(retry_config))