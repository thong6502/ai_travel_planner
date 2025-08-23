from config.logger import logger_hook
from typing import Optional
from langchain_core.tools import tool
from models.hotel import HotelSearchRequest
from loguru import logger

@tool
def kayak_hotel_url_generator(
    destination: str, check_in: str, check_out: str, adults: int = 1, children: int = 0, rooms: int = 1, sort: str = "recommended"
) -> str:
    """
    Generates a Kayak URL for hotel searches.

    Parameters:
    - destination (str): The destination city or area (e.g., "Berlin", "City Center, Singapore", "Red Fort, Delhi", "Huế, Việt-Nam").
    - check_in (str): The date of check-in in the format 'YYYY-MM-DD'.
    - check_out (str): The date of check-out in the format 'YYYY-MM-DD'.
    - adults (int): The number of adults, defaults to 1.
    - children (int): The number of children, defaults to 0.
    - rooms (int): The number of rooms, defaults to 1.
    - sort (str): The sort order, can be one of "recommended", "distance", "price", or "rating". Defaults to "recommended".

    Returns:
    - str: The generated Kayak URL for the hotel search.

    Example Input:
    kayak_hotel_url_generator(destination="New York", check_in="2025-12-24", check_out="2025-12-26", adults=2, rooms=1, sort="price")

    Example Output:
    "https://www.kayak.com/hotels/Huế,Việt-Nam/2025-12-24/2025-12-26/2adults?currency=USD&sort=price_a"
    """
    request = HotelSearchRequest(
        destination=destination,
        check_in=check_in,
        check_out=check_out,
        adults=adults,
        children=children,
        rooms=rooms,
        sort=sort)

    logger.info(f"Request: {request}")

    logger.info(f"Generating Kayak URL for {destination} on {check_in} to {check_out}")
    URL = f"https://www.kayak.com/hotels/{destination}/{check_in}/{check_out}"
    URL += f"/{adults}adults"
    if children > 0:
        URL += f"/{children}children"

    if rooms > 1:
        URL += f"/{rooms}rooms"


    URL += "?currency=USD"
    if sort.lower() == "price":
        URL += "&sort=price_a"
    elif sort.lower() == "rating":
        URL += "&sort=userrating_b"
    elif sort.lower() == "distance":
        URL += "&sort=distance_a"
    logger.info(f"URL: {URL}")
    return URL

# print(kayak_hotel_url_generator(destination="Huế,Việt-Nam", check_in="2025-12-24", check_out="2025-12-26", adults=2, rooms=1, sort="recommended"))
