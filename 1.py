import os
import requests
from langchain_core.tools import tool
from typing import Dict, Any, Optional

# --- CÀI ĐẶT API KEY ---
# Tốt nhất là lưu key vào biến môi trường và đọc từ đó
AVIATIONSTACK_API_KEY = os.environ.get("AVIATIONSTACK_API_KEY", "YOUR_AVIATIONSTACK_API_KEY")

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
    flight_date = flight_info.get('flight_date') # Sẽ là None nếu không có

    # --- Kiểm tra đầu vào ---
    if not dep_iata or not arr_iata:
        return {"error": "Dictionary đầu vào phải chứa cả 'dep_iata' và 'arr_iata'."}

    # URL và các tham số cơ bản của API
    API_URL = "http://api.aviationstack.com/v1/flights"
    params = {
        'access_key': AVIATIONSTACK_API_KEY,
        'dep_iata': dep_iata,
        'arr_iata': arr_iata,
        'limit': 10
    }

    # Nếu có ngày bay, thêm vào tham số
    if flight_date:
        params['flight_date'] = flight_date

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

# --- Ví dụ cách sử dụng ---
if __name__ == '__main__':
    if AVIATIONSTACK_API_KEY == "YOUR_AVIATIONSTACK_API_KEY":
        print("Vui lòng thay thế 'YOUR_AVIATIONSTACK_API_KEY' bằng khóa API thật của bạn.")
    else:
        # Tạo dictionary chứa thông tin chuyến bay cần tìm
        search_query = {
            "dep_iata": "HAN", 
            "arr_iata": "SGN", 
            "flight_date": "2025-08-22"
        }
        
        print(f"Đang tìm chuyến bay với thông tin: {search_query}")
        
        # Gọi công cụ với dictionary đó
        # Khi dùng với Agent, Agent sẽ tự động tạo và truyền dict này vào.
        # .invoke() của LangChain thường yêu cầu đầu vào là một dict, 
        # trong đó key là tên tham số của hàm.
        flights_data = get_flights.invoke({"flight_info": search_query})
        
        if "error" in flights_data:
            print(f"Lỗi khi tìm chuyến bay: {flights_data['error']}")
        elif flights_data and flights_data.get("data"):
            print(f"Tìm thấy {len(flights_data['data'])} chuyến bay:")
            for flight in flights_data['data']:
                flight_info_str = (
                    f"- Chuyến bay {flight['flight']['iata']} của hãng {flight.get('airline', {}).get('name', 'N/A')}, "
                    f"cất cánh lúc {flight['departure']['scheduled']}, "
                    f"hạ cánh lúc {flight['arrival']['scheduled']}"
                )
                print(flight_info_str)
        else:
            print("Không tìm thấy chuyến bay nào hoặc có lỗi xảy ra.")