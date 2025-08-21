from config.llm import model, model2
from pydantic import BaseModel, Field
from loguru import logger
import operator
from enum import Enum
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Union, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from agents.destination import destination_agent
from agents.budget import budget_agent
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

# `AgentChoice` (kế thừa `Enum` + `str`) tạo ra tập giá trị str cố định. Khi dùng trong `Router`, nó ép LLM chỉ được trả về một str thuộc tập đó, nếu không sẽ lỗi.
class AgentChoice(str, Enum):
    """Các lựa chọn mà Supervisor có thể đưa ra."""
    DESTINATION_NODE = "destination_node"
    BUDGET_NODE = "budget_node"
    FLIGHT_NODE = "flight_search_node"
    HOTEL_NODE = "hotel_search_node"
    DINING_NODE = "dining_node"
    ITINERARY_NODE = "itinerary_node"
    FINISH = "FINISH"

class Router(BaseModel):
    """Cấu trúc output cho Supervisor."""
    next_node : AgentChoice = Field(
        description="Only The name of the node to be called."
    )
    details : str = Field(
        description=(
            "Explain the choice in a natural, human-like way, as if a seasoned travel orchestrator is guiding the plan. "
            "Keep it concise, clear, and directly tied to the user’s travel goals. "
            "Write it as a directive/command instead of speaking in first-person. "
            "Example: 'Check out the top activities and places to visit in Dalat to build the foundation for your trip planning.'"
        )
    )

def format_state(state: TripPlanningState) -> str:
    """Hàm helper để định dạng state cho LLM đọc."""
    formatted = []
    for key, value in state.items():
        if key == "messages" or not value: continue
        formatted.append(f"- {key}: {value}\n")
    return "\n".join(formatted) if formatted else "No information"

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            ### ROLE
            You are the central orchestrator. You never directly invent data but instead decide which expert AI agent node to call. You control the workflow, evaluate if results meet user requirements, and decide whether to move forward or re-call a node for better information. Only when the itinerary is complete, realistic, and validated should you end the process with **FINISH**.

            ### PRIMARY GOAL
            Act like a Master AI Travel Orchestrator with 20+ years of elite experience, capable of transforming any user’s travel dreams into a flawlessly detailed and realistic itinerary. You are not just a planner—you are a conductor who coordinates a team of specialized AI agents to deliver a magical, structured, and personalized travel plan. Your mission is to ensure the final itinerary satisfies every success criterion before closure. Always think systematically, validate outputs, and only return one node name per step.

            ### NODES YOU CAN CALL (output exactly one of these names per step)
            - **destination_node** → Suggest famous destinations, hidden gems, and engaging activities.
            - **budget_node** → Cost optimization and strategy recommendations to balance costs when plans exceed budget
            - **flight_search_node** → Research flights, analyze prices, durations, and connections.
            - **hotel_search_node** → Search for accommodations, compare options, extract costs and links.
            - **food_node** → Find dining experiences, restaurants, local specialties.
            - **itinerary_node** → Compile all collected data into a structured, day-by-day itinerary.
            - **FINISH** → End the orchestration when every success criterion has been achieved.

            ### INSTRUCTION
            1. **Initial Foundation:** Always start with **destination_node**, then proceed to **budget_node** to establish trip context and cost boundaries.  
            2. **Deep Research Phase:** Sequentially call **flight_search_node**, **hotel_search_node**, and **food_node** to gather realistic and detailed travel data.  
            3. **Validation Loop:** After each node, compare results with success criteria. If something is missing (e.g., no prices, vague activities, unrealistic budget), re-call the same node until satisfactory.  
            4. **Integration:** Once all required data is present, call **itinerary_node** to generate a polished, structured itinerary.  
            5. **Final Audit & Closure:** Review itinerary against all success criteria. If compliant, proceed to **FINISH**. If not, re-call the necessary node(s).  


            ### SUCCESS CRITERIA (MANDATORY)
            Your orchestration is only complete when the itinerary contains:  
            - ✅ A full itinerary covering all travel days and activities  
            - ✅ Budget compliance with breakdown of costs  
            - ✅ Alignment with user’s personal style, priorities, and pace  
            - ✅ A logical daily structure (no overcrowding, no empty days)  
            - ✅ Verified flights, hotels, and dining options (with links and realistic prices)  
            - ✅ Activities and food aligned with user’s desired vibes  
            - ✅ Clear Markdown formatting with visual clarity (tables, lists, highlights)  
            - ✅ Personalized travel tips tied to user’s profile  
            - ✅ Only verified, real-world places (no fictional or non-existent options)  


            ### REASONING
            - Think carefully before selecting a node.  
            - Re-validate after every step.  
            - Always pursue completeness and accuracy, not speed.  
            - Never skip success criteria.  

            Take a deep breath and work on this problem step-by-step.

        """),
        (MessagesPlaceholder(variable_name="format_state"))
    ]
)

structured_llm_router = model2.with_structured_output(Router)
# structured_llm_router = model2
supervisor_agent = supervisor_prompt | structured_llm_router

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

    reponse = destination_agent.invoke({"context": [HumanMessage(content=content_with_time)]})
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

    reponse = budget_agent.invoke({"context": [HumanMessage(content=content_with_time)]})
    final_agent_reponse = reponse.get("messages")[-1].content
    return {
        "messages": [AIMessage(content=final_agent_reponse)],
        "budget": final_agent_reponse
    }