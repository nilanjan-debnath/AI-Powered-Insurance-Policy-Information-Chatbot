from .tool_list import tools
from .history import get_history_by_id, save_history

from langchain.agents import AgentExecutor
from langchain_cohere import ChatCohere
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import os
import logging
from dotenv import load_dotenv

load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{preamble}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

try:
    llm = ChatCohere(model="command-r-plus")
except Exception as e:
    logging.error(
        f"FILE: app/agent.py INFO: Failed to initialize llm. DETAILS: {e}",
        exc_info=True,
    )
    raise e

try:
    agent = create_cohere_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        # verbose=True,
        handle_parsing_errors=True,
        # return_intermediate_steps=True,
    )
except Exception as e:
    logging.error(
        f"FILE: app/agent.py INFO: Failed to create agent. DETAILS: {e}", exc_info=True
    )
    raise e


tool_descriptions = "\n".join(
    [f"- Tool Name: {tool.name}\n  Description: {tool.description}" for tool in tools]
)
preamble = ""
preamble_file = "data/document/preamble.txt"
try:
    with open(preamble_file, "r") as f:
        preamble_template = f.read()
        preamble = preamble_template.replace("#tool_descriptions#", tool_descriptions)
except Exception as e:
    logging.warning(
        f"FILE: app/agent.py INFO: Couldn't load Preamble file {preamble_file}. DETAILS: {e}",
        exc_info=True,
    )
    preamble = "Default system prompt: You are a helpful assistant."


def run_agent(query: str, id: str) -> str:
    logging.debug(
        f"FILE: app/agent.py INFO: Running agent for conversation ID: {id} with query: '{query}'"
    )
    chat_history = get_history_by_id(id)
    logging.debug(
        f"FILE: app/agent.py INFO: Loaded chat history for {id}: {chat_history}"
    )

    try:
        response = agent_executor.invoke(
            {"preamble": preamble, "input": query, "chat_history": chat_history}
        )
        output = response.get("output", "Sorry, I could not process that.")
        logging.debug(f"FILE: app/agent.py INFO: Agent response for {id}: '{output}'")

        updated_history = list(chat_history)
        updated_history.append(HumanMessage(content=query))
        updated_history.append(AIMessage(content=output))

        save_history(id, updated_history)
        logging.debug(f"FILE: app/agent.py INFO: Saved updated history for {id}")
        return output

    except Exception as e:
        logging.error(
            f"FILE: app/agent.py INFO: Error during agent execution for conversation {id}. DETAILS: {e}",
            exc_info=True,
        )
        return f"Sorry, an error occurred while processing your request: {e}"
