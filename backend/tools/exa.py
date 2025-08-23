from langchain_exa import ExaFindSimilarResults, ExaSearchResults
import os
from dotenv import load_dotenv

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
