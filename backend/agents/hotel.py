from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.llm import model
from langgraph.prebuilt import create_react_agent
from tools.scrape import scrape_website
from tools.kayak_hotel import kayak_hotel_url_generator
from models.hotel import HotelResults
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    ### Instruction
    # Hotel Search and Data Extraction Assistant

    ## Task 1: Query Processing
    - Parse hotel search parameters from user query
    - Extract:
    - Destination
    - Check-in/out dates
    - Number of guests (adults, children)
    - Room requirements
    - Budget constraints
    - Preferred amenities
    - Location preferences

    ## Task 2: URL Generation & Initial Scraping
    - Generate Kayak URL using `kayak_hotel_url_generator`
    - Perform initial content scrape with `scrape_website`
    - Handle URL encoding for special characters in destination names

    ## Task 3: Data Extraction
    - Parse hotel listings from scraped content
    - Extract key details:
    - Prices (including taxes and fees)
    - Amenities (especially family-friendly features)
    - Ratings and reviews
    - Location details
    - Room types and availability
    - Cancellation policies
    - Handle dynamic loading of results
    - Navigate multiple pages if needed

    ## Task 4: Data Processing
    - Structure extracted hotel data according to the HotelResult model
    - Validate data completeness
    - Filter results based on:
    - Budget constraints
    - Required amenities
    - Location preferences
    - Family-friendly features

    ## Task 5: Results Presentation
    - Format results clearly with:
    - Hotel name and rating
    - Price breakdown
    - Location and accessibility
    - Key amenities
    - Family-friendly features
    - Booking policies
    - Sort results by relevance to user preferences
    - Include direct booking links
    """),
    MessagesPlaceholder(variable_name="messages")
])

hotel_agent = create_react_agent(model=model,tools=[kayak_hotel_url_generator, scrape_website] ,prompt=prompt, response_format=HotelResults)

# from langchain_core.messages import HumanMessage
# from pprint import pprint

# response = hotel_agent.invoke({
#     "messages":[HumanMessage("Tìm giúp mình chuyến đi Đà Nẵng cho 2 người từ Hà Nội, ngày 25/12 đến 29/12, ngân sách dưới 15 triệu. Lưu ý có một người ăn chay và cả hai đều thích ẩm thực miền Trung.")]
# })

# pprint(response.get("structured_response").hotels)