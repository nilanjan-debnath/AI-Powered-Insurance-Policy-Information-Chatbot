from db.pinecone_db import get_vector_store

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging


class QueryInput(BaseModel):
    query: str = Field(
        description="The search term or phrase to find relevant documents"
    )


@tool("insurance_policy_knowledge_search", args_schema=QueryInput, return_direct=False)
def insurance_policy_knowledge_search(query: str) -> list:
    """Searches and retrieves relevant information exclusively from the official [Your Insurance Company Name] knowledge base. This knowledge base contains detailed documents about all our insurance policies (health, life, auto, home), including specific coverage options, benefits, limitations, exclusions, premium details, and step-by-step claim processes.
    Args:
        query(str): A string containing the user's specific question or a reformulated query focusing on insurance policy details, coverage, premiums, or claims.

    Return:
        results(list): A list of documents containing the most relevant excerpts or summaries extracted directly from the official company documents that match the input query.

    Use this tool whenever a user asks about:
    - Specific details of any [Your Insurance Company Name] insurance policy.
    - What is covered or excluded under a particular plan.
    - Information about insurance premiums, costs, or payment options.
    - How to file a claim or the status of a claim process.
    - Eligibility criteria for policies.
    - Comparisons between different [Your Insurance Company Name] policies.

    Do NOT use this tool for greetings, general conversation, or questions unrelated to [Your Insurance Company Name]'s insurance products and procedures.
    """
    logging.info(
        f"FILE: app/tool_list.py INFO: Performing similarity search for query: '{query}"
    )
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search(query, k=5)
        logging.info(
            f"FILE: app/tool_list.py INFO: Found {len(results)} relevant documents."
        )
    except Exception as e:
        results = []
        logging.error(
            f"FILE: app/tool_list.py INFO: Error during similarity search. DETAILS: {e}",
            exc_info=True,
        )
    return results


tools = [insurance_policy_knowledge_search]
