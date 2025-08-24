from loguru import logger
import operator
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from pprint import pprint
from langchain_core.messages import HumanMessage

from agents.destination import destination_agent
from agents.budget import budget_agent
from agents.supervisor import supervisor_agent
from agents.food import dining_agent
from agents.itinerary import itinerary_agent
from agents.flight import flight_agent
from agents.hotel import hotel_agent

class TripPlanningState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_input: str
    destination: str
    budget: dict
    flight_options: List[any]
    hotel_options: List[any]
    dining: str
    itinerary: str


def format_state(state: TripPlanningState) -> str:
    """Hàm helper để định dạng state cho LLM đọc."""
    formatted = []
    for key, value in state.items():
        if key == "messages" or not value: continue
        formatted.append(f"- {key}: {value}\n")
    return "\n".join(formatted) if formatted else "No information"

def supervision(state: TripPlanningState) -> str:
    print("---- CALL SUPERVISION ----")
    print(f"Current State: \n{format_state(state)}")

    response = supervisor_agent.invoke({
        "messages": state["messages"]
    })

    structured_response = response.get("structured_response")
    # Lưu lại chi tiết reasoning của supervisor vào messages
    state["messages"].append(HumanMessage(content=structured_response.details))

    # Trả về tên node kế tiếp (ở dạng string)
    return structured_response


def destination_node(state: TripPlanningState) -> dict:
    print("---- CALL DESTINATION NODE ----")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lass_supervision_mess  : HumanMessage = state["messages"][-1]
    
    content_with_time = f"### Current Plan: {format_state(state)}\n### Require: {lass_supervision_mess.content}\n\n(The request was processed at: {now_str})"

    reponse = destination_agent.invoke({"messages": [HumanMessage(content=content_with_time)]})
    final_agent_reponse = reponse.get("messages")[-1].content
    return {
        "messages": [AIMessage(content=final_agent_reponse)],
        "destination": final_agent_reponse
    }
    # 8.000 -> 15.000 token

def budget_node(state: TripPlanningState) -> dict:
    print("---- CALL BUDGET NODE ----")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lass_supervision_mess  : HumanMessage = state["messages"][-1]
    
    content_with_time = f"### Current Plan: {format_state(state)}\n### Require: {lass_supervision_mess.content}\n\n(The request was processed at: {now_str})"

    reponse = budget_agent.invoke({"messages": [HumanMessage(content=content_with_time)]})
    final_agent_reponse = reponse.get("messages")[-1].content
    return {
        "messages": [AIMessage(content=final_agent_reponse)],
        "budget": final_agent_reponse
    }

def hotel_node(state: TripPlanningState) -> dict:
    print("---- CALL HOTEL NODE -----")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lass_supervision_mess  : HumanMessage = state["messages"][-1]
    
    content_with_time = f"### Current Plan: {format_state(state)}\n### Require: {lass_supervision_mess.content}\n\n(The request was processed at: {now_str})"

    reponse = hotel_agent.invoke({"messages": [HumanMessage(content=content_with_time)]})
    final_agent_reponse = reponse.get("messages")[-1].content
    return {
        "messages": [AIMessage(content=final_agent_reponse)],
        "hotel_options": reponse.get("structured_response")
    }
    # 30.000 - 40.000 token
    # 'hotel_options': HotelResults(hotels=[HotelResult(hotel_name='Ha Giang Vegetarian Homestay', price='240,000 VND per night', rating='8.3', address='Ha Giang, Vietnam', amenities=['Free breakfast', 'Free cancellation', 'Vegetarian-friendly'], description='A cozy homestay offering vegetarian meals and a friendly atmosphere.', url='https://www.kayak.com/hotels/Ha-Giang-Vegetarian-Homestay,Ha-Giang-p2631332-h1072082236-details/2025-10-02/2025-10-06/2adults?psid=uHCEClmeOr&pm=nightly-base')])

def flight_node(state: TripPlanningState) -> dict:
    print("---- CALL FLIGHT NODE ----")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lass_supervision_mess  : HumanMessage = state["messages"][-1]
    
    content_with_time = f"### Current Plan: {format_state(state)}\n### Require: {lass_supervision_mess.content}\n\n(The request was processed at: {now_str})"

    reponse = flight_agent.invoke({"messages": [HumanMessage(content=content_with_time)]})
    final_agent_reponse = reponse.get("messages")[-1].content
    return {
        "messages": [AIMessage(content=final_agent_reponse)],
        "flight_options": reponse.get("structured_response")
    }
    # 10.000 -> 15.000 token
    # 'flight_options': FlightResults(flights=[FlightResult(price='4870080 VND', airline='Vietravel Airlines', departure_time='6:25 AM on Thu, Oct 2', arrival_time='7:45 AM on Thu, Oct 2', duration='1 hr 20 min', stops=0), FlightResult(price='5561280 VND', airline='Vietjet', departure_time='5:30 AM on Mon, Oct 6', arrival_time='6:50 AM on Mon, Oct 6', duration='1 hr 20 min', stops=0)]),

def dining_node(state: TripPlanningState) -> dict:
    print("---- CALL DINING NODE ----")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lass_supervision_mess  : HumanMessage = state["messages"][-1]
    
    content_with_time = f"### Current Plan: {format_state(state)}\n### Require: {lass_supervision_mess.content}\n\n(The request was processed at: {now_str})"

    reponse = dining_agent.invoke({"messages": [HumanMessage(content=content_with_time)]})
    final_agent_reponse = reponse.get("messages")[-1].content
    return {
        "messages": [AIMessage(content=final_agent_reponse)],
        "dining": final_agent_reponse
    }
    # 13.000 -> 20.000 token

def itinerary_node(state: TripPlanningState) -> dict:
    print("---- CALL ITINARARY NODE ----")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lass_supervision_mess  : HumanMessage = state["messages"][-1]
    
    content_with_time = f"### Current Plan: {format_state(state)}\n### Require: {lass_supervision_mess.content}\n\n(The request was processed at: {now_str})"

    reponse = itinerary_agent.invoke({"messages": [HumanMessage(content=content_with_time)]})
    final_agent_reponse = reponse.get("messages")[-1].content
    return {
        "messages": [AIMessage(content=final_agent_reponse)],
        "itinerary": final_agent_reponse
    }
    # 5.000 -> 10.000 token



# --------------------------------------------
from mock_data import mock_state

pprint(itinerary_node(mock_state))