from firecrawl import FirecrawlApp
import os
from langchain_core.tools import tool
from loguru import logger
from config.logger import logger_hook
from dotenv import load_dotenv

load_dotenv()
app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))


@tool
def scrape_website(url: str) -> str:
    """Scrape a website and return the markdown content.

    Args:
        url (str): The URL of the website to scrape.

    Returns:
        str: The markdown content of the website.

    Example:
        >>> scrape_website("https://www.google.com")
        "## Google"
    """
    logger.info(f"Start scapre website: {url}")
    scrape_status = app.scrape(
        url,
        formats=["markdown"],
        wait_for=30000,
        timeout=60000,
    )
    return scrape_status.markdown

# from pprint import pprint
# pprint(scrape_website.invoke({"url":"https://www.kayak.com/hotels/Hue,-Thua-thien-hue-c28212-lHUI/2025-12-24/2025-12-26/2adults;map?sort=distance_a"}))