from langchain_exa import ExaFindSimilarResults, ExaSearchResults
import os
from loguru import logger
from typing import Union
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Initialize exa tools
exa_search = ExaSearchResults(
    exa_api_key=os.environ.get("EXA_API_KEY"),
    # verbose=True
)

exa_find_similar = ExaFindSimilarResults(
    exa_api_key=os.environ.get("EXA_API_KEY"),
    # verbose=True
)


@tool
def CustomExaSearch(query: str) -> Union[list[dict] | str]:
    """Searches the web for information on a given topic using the Exa Search API.

    This tool is ideal for finding up-to-date information, articles, or data
    related to a user's query.

    Args:
        query (str): Input should be an Exa-optimized query. 

    Returns:
        Output is a JSON array of the query results
    """
    # Các tham số mặc định được truyền trực tiếp vào đây

    logger.info(
        f"Exa search with query {query}"
    )
    return exa_search.invoke({
        "query": query, 
        "num_results": 2, 
        "text_contents_options": {"max_characters": 3000}
    })


@tool
def CustomExaFindSimilar(url: str) -> Union[list[dict] | str]:
    """Finds web pages with content similar to a given URL using the Exa API.

    Use this tool when you have a URL of a relevant article or source and
    want to discover other pages that discuss similar topics.

    Args:
        url (str): The full URL of the source page to find similar content for.
            For example: "https://www.example.com/article/on-topic-x".

    Returns:
        Output is a JSON array of the query results
    """
    # Sửa lại tham số từ "query" thành "url" để khớp với yêu cầu của công cụ
    logger.info(
        f"Exa find similar with query {url}"
    )
    return exa_find_similar.invoke({
        "url": url, 
        "num_results": 2, 
        "text_contents_options": {"max_characters": 3000}
    })