from config.llm import model
from loguru import logger

from typing import TypedDict

class TravelState(TypedDict):
    query: str
    destination: str
    flights: list
    hotels: list
    dining: list
    budget: dict
    itinerary: str
    result: str
