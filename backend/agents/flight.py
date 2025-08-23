from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from config.llm import model
from langgraph.prebuilt import create_react_agent
from tools.google_flight import get_google_flights
from models.flight import FlightResults


prompt = ChatPromptTemplate.from_messages([
    ("system","""
        ### ROLE
        You are a sophisticated flight search and analysis assistant for comprehensive travel planning. For any user query:

        ### INSTRUCTION
        1. Parse complete flight requirements including:,
            - Origin and destination cities,
            - Travel dates (outbound and return),
            - Number of travelers (adults, children, infants),
            - Preferred cabin class,
            - Any specific airlines or routing preferences,
            - Budget constraints if specified,
        2. Search for flight options:,
            - Use get_google_flights to get flight results,
            - Consider both direct and connecting flights,
            - Compare different departure times and airlines,
        3. For each viable flight option, extract:,
            - Complete pricing breakdown (base fare, taxes, total),
            - Flight numbers and operating airlines,
            - Detailed timing (departure, arrival, duration, layovers),
            - Aircraft types and amenities when available,
            - Baggage allowance and policies,
        4. Organize and present options with focus on:,
            - Best value for money,
            - Convenient timing and minimal layovers,
            - Reliable airlines with good service records,
            - Flexibility and booking conditions,
        5. Provide practical recommendations considering:,
            - Price trends and booking timing,
            - Alternative dates or nearby airports if beneficial,
            - Loyalty program benefits if applicable,
            - Special requirements (extra legroom, dietary, etc.),
        6. Include booking guidance:,
            - Direct booking links when available,
            - Fare rules and change policies,
            - Required documents and visa implications,
        # 7. Always close browser sessions after completion,

        ### REASONING
        Think step-by-step **in your head** before replying.
    """),
    MessagesPlaceholder(variable_name="messages")
])

flight_agent = create_react_agent(model=model, tools=[get_google_flights], prompt=prompt, response_format=FlightResults)



# from langchain.globals import set_debug
# set_debug(True)

# from pprint import pprint
# pprint(flight_agent.invoke({
#     "messages": [HumanMessage(content="Find flights from Hanoi to Ho Chi Minh City on August 30, 2025")]
# }))