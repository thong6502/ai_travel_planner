import os
from config.llm import model
# from agents.team import TravelState
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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
distination_agent = create_react_agent(model=model, tools=[exa_search, exa_find_similar], prompt=prompt)

now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
response = distination_agent.invoke({
    "messages": [
        HumanMessage(content=f"Tìm địa điểm du lịch trên ba vì và phản hồi nội dung lại bằng tiếng việt. Thời gian hỏi hiện tại là {now_str}")
    ]
}, config={"recursion_limit": 4, "return_intermediate_steps": True})

from pprint import pprint
pprint(response.get("messages")[-1].content)


# print(response)

# def distination_node(state: TravelState) -> TravelState:
#     human_message = state.get("query", "").strip()
#     now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

#     distination_agent.invoke(f"")
