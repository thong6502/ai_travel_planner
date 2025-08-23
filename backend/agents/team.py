from config.llm import model, model2
from pydantic import BaseModel, Field
from loguru import logger
import operator
from enum import Enum
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Union, List, Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from agents.destination import destination_agent
from agents.budget import budget_agent
from agents.supervisor import supervisor_agent
from datetime import datetime

class TripPlanningState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_input: str
    destination: str
    budget: dict
    flight_options: List[dict]
    hotel_options: List[dict]
    dining_options: List[dict]
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
        "format_state": [HumanMessage(content=format_state(state))]
    })

    # Lưu lại chi tiết reasoning của supervisor vào messages
    state["messages"].append(HumanMessage(content=response.details))

    # Trả về tên node kế tiếp (ở dạng string)
    return response.next_node.value

def destination_node(state: TripPlanningState) -> dict:
    print("---- CALL DESTINATION NODE ----")

    messages = state.get("messages", [])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lass_user_messages = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    
    content_with_time = f"### Current Plan: {format_state(state)}\n### Require: {lass_user_messages.content}\n\n(The request was processed at: {now_str})"

    reponse = destination_agent.invoke({"messages": [HumanMessage(content=content_with_time)]})
    final_agent_reponse = reponse.get("messages")[-1].content
    return {
        "messages": [AIMessage(content=final_agent_reponse)],
        "destination": final_agent_reponse
    }

def budget_node(state: TripPlanningState) -> dict:
    print("---- CALL BUDGET NODE ----")

    messages = state.get("messages", [])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lass_user_messages = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    
    content_with_time = f"### Current Plan: {format_state(state)}\n### Require: {lass_user_messages.content}\n\n(The request was processed at: {now_str})"

    reponse = budget_agent.invoke({"messages": [HumanMessage(content=content_with_time)]})
    final_agent_reponse = reponse.get("messages")[-1].content
    return {
        "messages": [AIMessage(content=final_agent_reponse)],
        "budget": final_agent_reponse
    }