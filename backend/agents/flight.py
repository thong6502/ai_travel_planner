from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from config.llm import model
import requests
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

load_dotenv()

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
    MessagesPlaceholder(variable_name="context")
])

@tool
def get_flights(flight_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tìm kiếm các chuyến bay dựa trên thông tin được cung cấp trong một dictionary.
    Dictionary đầu vào phải chứa các key: 'dep_iata' (bắt buộc), 'arr_iata' (bắt buộc), 
    và 'flight_date' (tùy chọn, định dạng YYYY-MM-DD).
    Ví dụ: {"dep_iata": "HAN", "arr_iata": "SGN", "flight_date": "2025-08-22"}
    """

    # Trích xuất thông tin từ dictionary đầu vào
    dep_iata = flight_info.get('dep_iata')
    arr_iata = flight_info.get('arr_iata')
    flight_date = flight_info.get('flight_date')

    # --- Kiểm tra đầu vào ---
    if not dep_iata or not arr_iata:
        return {"error": "Dictionary đầu vào phải chứa cả 'dep_iata' và 'arr_iata'."}
    
    # URL và các tham số cơ bản của API
    API_URL = "http://api.aviationstack.com/v1/flights"
    params = {
        'access_key': os.getenv("AVIATIONSTACK_FLIGHT_API"),
        'dep_iata': dep_iata,
        'arr_iata': arr_iata,
        'flight_date':flight_date,
        'limit': 10
    }

        # --- Gọi API và xử lý lỗi ---
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error occurred: {http_err}", "details": response.text}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"An error occurred: {req_err}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}