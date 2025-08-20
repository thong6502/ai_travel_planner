import os
from config.llm import model
from agents.team import TripPlanningState
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import HumanMessage, AIMessage ,SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langchain_exa import ExaFindSimilarResults, ExaSearchResults

load_dotenv()

# Initialize exa tools
exa_search = ExaSearchResults(
    exa_api_key=os.environ.get("EXA_API_KEY"),
    max_results=5
)

exa_find_similar = ExaFindSimilarResults(
    exa_api_key=os.environ.get("EXA_API_KEY"),
    max_results=5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    You are a destination research agent that focuses on recommending mainstream tourist attractions 
    and classic experiences that most travelers would enjoy. You prioritize well-known landmarks and 
    popular activities while keeping recommendations general and widely appealing.

    Instructions:
    1. Focus on mainstream attractions with thoughtful guidance:
        - Famous landmarks and monuments
        - Popular tourist spots
        - Well-known museums
        - Classic shopping areas
        - Common tourist activities

    2. Guide visitors with simple reasoning:
        - Suggest crowd-pleasing activities
        - Focus on family-friendly locations
        - Recommend proven tourist routes
        - Include popular photo spots

    3. Present clear attraction information:
        - Simple description
        - General location
        - Regular opening hours
        - Standard entrance fees
        - Typical visit duration
        - Basic visitor tips

    4. Organize information logically:
        - Main attractions first
        - Common day trips
        - Standard tourist areas
        - Popular activities

    Expected Output (in Markdown):
    # Tourist Guide
    ## Main Attractions
    List of most popular tourist spots

    ## Common Activities
    Standard tourist activities and experiences

    ## Popular Areas
    Well-known districts and neighborhoods

    ## Basic Information
    - General visiting tips
    - Common transportation options
    - Standard tourist advice
    """),
    MessagesPlaceholder(variable_name="messages"),
])

# Create agent with bold tools
destination_agent = create_react_agent(model=model, tools=[exa_search, exa_find_similar], prompt=prompt)

def destination_node(state: TripPlanningState) -> dict:
    print("---- CALL DESTINATION NODE ----")

    messages = state.get("messages", [])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lass_user_messages = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)

    if not lass_user_messages:
        return {
            "messages": [AIMessage(content="Error: Cannot find a request from user to process")]
        }
    
    content_with_time = f"{lass_user_messages.content}\n\n(The request was processed at: {now_str})"
    reponse = destination_agent.invoke({"messages": [HumanMessage(content=content_with_time)]})
    final_agent_reponse = reponse.get("messages")[-1].content
    return {"messages": [AIMessage(content=final_agent_reponse)]}

if __name__ == "__main__":

    from pprint import pprint

    # Tạo một state giả lập để test
    initial_state: TripPlanningState = {
        "messages": [
            HumanMessage(content="Tìm địa điểm du lịch ở Ba Vì.")
        ]
    }
    result_update = destination_node(initial_state)

    print("\n--- KẾT QUẢ CẬP NHẬT STATE ---")
    pprint(result_update)