import os

import logging
import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentInterface

from agent import WarehouseManagerAgent
from agent_executor import WarehouseManagerExecutor

from dotenv import load_dotenv

from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from starlette.applications import Starlette
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes
)

load_dotenv()

HOST = os.getenv("HOTS", "localhost")
PORT = int(os.getenv("PORT", 8080))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    
    capabilities = AgentCapabilities(streaming=True)
    skill_availability = AgentSkill(
        id="abc",
        name="check_availability",
        description="Check Stock Availability in Warehouse",
        tags=["Warehouse", "Availability"],
        examples=["What is the avaialability of this item B123"]
    )

    skill_reservation = AgentSkill(
        id="def",
        name="make_reservation",
        description="Make a reservation for an item in the warehouse",
        tags=["Warehouse", "Reservation"],
        examples=["Reserve 3 quantities of item B123"]
    )

    agent_interface = AgentInterface(
        url=f"http://{HOST}:{PORT}/",
        protocol_binding="JSONRPC",
        protocol_version="1.0.0"
    )
    
    agent_card = AgentCard(
        name="warehouse_manager_agent",
        description="A Warehouse manager agent that can check stock availability and reserve items",
        version="1.0.0",
        skills=[skill_availability, skill_reservation],
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        supported_interfaces=[agent_interface]
    ) 

    adk_agent = WarehouseManagerAgent().get_agent()

    runner = Runner(
        app_name=agent_card.name,
        agent=adk_agent,
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        memory_service=InMemoryMemoryService()
    )

    agent_executor = WarehouseManagerExecutor(runner)

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/"))
    routes.extend(create_rest_routes(request_handler))

    app = Starlette(routes=routes)

    logger.info(f"Starting manual custom server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == "__main__":
    main()