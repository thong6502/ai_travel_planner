from fast_flights import FlightData, Passengers, Result, get_flights
from typing import Literal
from loguru import logger
from langchain_core.tools import tool
# from config.logger import logger_hook


@tool
def get_google_flights(
    departure: str,
    destination: str,
    date: str,
    trip: Literal["one-way", "round-trip"] = "one-way",
    adults: int = 1,
    children: int = 0,
    cabin_class: Literal["first", "business", "premium-economy", "economy"] = "economy",
) -> Result:
    """
    Get flights from Google Flights.

    Parameters:
    - departure (str): The IATA departure airport code (e.g., "SFO").
    - destination (str): The IATA destination airport code (e.g., "JFK").
    - date (str): The date of the flight in the format 'YYYY-MM-DD'.
    - trip (Literal["one-way", "round-trip"]): The type of trip, defaults to "one-way".
    - adults (int): The number of adults, defaults to 1.
    - children (int): The number of children, defaults to 0.
    - cabin_class (Literal["first", "business", "premium-economy", "economy"]): The cabin class, defaults to "economy".

    Returns:
    - List: A list containing the flight information found. Returns an empty list if an error occurs.

    Example Input:
    get_google_flights(departure="SFO", destination="JFK", date="2025-12-25", adults=2, trip="one-way")

    Example Output:
    [
        Flight(is_best=True,
                name='Alaska',
                departure='6:20 AM on Thu, Dec 25',
                arrival='3:05 PM on Thu, Dec 25',
                arrival_time_ahead='',
                duration='5 hr 45 min',
                stops=0,
                delay=None,
                price='₫7888116'),
        Flight(is_best=True,
                name='Alaska',
                departure='10:48 PM on Thu, Dec 25',
                arrival='7:29 AM on Fri, Dec 26',
                arrival_time_ahead='+1',
                duration='5 hr 41 min',
                stops=0,
                delay=None,
                price='₫9948642')
    ]
    """
    logger.info(
        f"Getting flights from Google Flights for {departure} to {destination} on {date}, trip({trip}), adults({adults}), children({children}), cabin_class({cabin_class})"
    )

    try:
        result: Result = get_flights(
            flight_data=[
                FlightData(date=date, from_airport=departure, to_airport=destination)
            ],
            trip=trip,
            seat=cabin_class,
            passengers=Passengers(
                adults=adults, children=children, infants_in_seat=0, infants_on_lap=0
            ),
            fetch_mode="fallback",
        )
        # logger.info(f"Flights found: {result.flights}")

        return result.flights
    except Exception as e:
        logger.error(f"Error getting flights from Google Flights: {e}")
        return []

if __name__ == "__main__":
    from pprint import pprint
    tool_input = {
        "departure": "SFO",
        "destination": "JFK",
        "date": "2025-12-25",
        "adults": 2,
        "trip": "one-way"
    }
    pprint(get_google_flights.args)
    #You can call the tool using a dictionary as input
    # pprint(get_google_flights.invoke(tool_input))