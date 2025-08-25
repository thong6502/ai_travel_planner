from typing import Literal
from pydantic import BaseModel, Field

AgentChoice = Literal[
    "destination_node",
    "budget_node",
    "flight_node",
    "hotel_node",
    "dining_node",
    "itinerary_node",
    "FINISH"
]

class supervisor_result(BaseModel):
    """Cấu trúc output cho Supervisor."""
    next_node : AgentChoice = Field(
        description="Only The name of the node to be called."
    )
    details : str = Field(
        description=(
            "Explain the choice in a natural, human-like way, as if a seasoned travel orchestrator is guiding the plan. Keep it concise, clear, and directly tied to the user’s travel goals. Write it as a directive/command instead of speaking in first-person. "
            "Example: 'Check out the top activities and places to visit in Dalat to build the foundation my trip.'"
        )
    )