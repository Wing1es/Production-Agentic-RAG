import os
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.runners import Runner, RunConfig
from google.adk.models import LiteLlm

# Load environment variables from .env
load_dotenv()

# Import tools from the local tools module
from warehouse_manager_agent.tools import check_warehouse_availability, reserve_warehouse_items

model = LiteLlm(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

agent = Agent(
    name="Warehouse_manager_agent",
    model=model,
    tools=[check_warehouse_availability, reserve_warehouse_items],
    description="You are a warehouse manager agent who can check items for availability and reserve them.",
    instruction=(
        "You are a warehouse manager agent. Your primary responsibility is to check and reserve inventory. "
        "Always verify the availability of items in the warehouses before making any reservations. "
        "Only reserve items when the entire order can be fulfilled, or if the user has confirmed they "
        "want a partial reservation. If stock is limited in a single warehouse, attempt to combine "
        "quantities from multiple warehouses. When suggesting tool calls, set final_answer to False. "
        "Once all necessary tools have returned their results and you are ready to respond to the user, "
        "set final_answer to True."
    )
)

warehouse_agent = agent
root_agent = agent