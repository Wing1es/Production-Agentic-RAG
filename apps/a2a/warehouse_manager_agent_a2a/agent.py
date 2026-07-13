import os
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.models import LiteLlm

load_dotenv()

from tools import check_warehouse_availability, reserve_warehouse_items

class WarehouseManagerAgent():

    def __init__(self):
        self.model = LiteLlm(
            model="groq/llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.agent = Agent(
            name="Warehouse_manager_agent",
            model=self.model,
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
    
    def get_agent(self) -> Agent:
        return self.agent