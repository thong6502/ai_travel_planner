from config.llm import model
from loguru import logger
import operator
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Union, List

class TripPlanningState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_input: str
    destination: str
    dates: str
    budget: dict
    flight_options: List[dict]
    hotel_options: List[dict]
    dining_options: List[dict]
    itinerary: str
    next_agent: str