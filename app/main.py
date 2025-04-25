from config_logging import set_logging_config
import streamlit as st
import uuid
import logging

from model.agent import run_agent, preamble
from model.history import (
    get_history_by_id,
    get_all_conversation_ids,
    create_new_conversation,
)
from langchain_core.messages import HumanMessage, AIMessage

set_logging_config()

st.set_page_config(page_title="SecureLife Insurance", layout="wide")
st.title("🤝 SecureLife Insurance AI Agent")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if "conversation_ids" not in st.session_state:
    st.session_state.conversation_ids = get_all_conversation_ids()


def load_conversation(conv_id):
    st.session_state.messages = get_history_by_id(conv_id)
    st.session_state.current_conversation_id = conv_id
    logging.info(f"FILE: app/main.py INFO: Switched to conversation: {conv_id}")


st.sidebar.title("Conversations")

if st.sidebar.button("➕ New Conversation"):
    new_conv_id = f"conv_{uuid.uuid4().hex[:8]}"
    create_new_conversation(new_conv_id)
    st.session_state.conversation_ids.append(new_conv_id)
    load_conversation(new_conv_id)
    st.rerun()

st.sidebar.markdown("---")


if not st.session_state.conversation_ids:
    st.sidebar.write("No conversations yet.")
else:
    try:
        current_index = st.session_state.conversation_ids.index(
            st.session_state.current_conversation_id
        )
    except (ValueError, TypeError):
        current_index = 0

    selected_conv_id = st.sidebar.radio(
        "Select Conversation:",
        options=st.session_state.conversation_ids,
        index=current_index,
        key=f"conv_selector_{st.session_state.current_conversation_id}",
    )

    if selected_conv_id != st.session_state.current_conversation_id:
        load_conversation(selected_conv_id)
        st.rerun()


if not st.session_state.current_conversation_id:
    st.info("Select a conversation or create a new one from the sidebar.")
else:
    st.subheader(f"Chatting in: `{st.session_state.current_conversation_id}`")

    for message in st.session_state.messages:
        with st.chat_message(
            "user" if isinstance(message, HumanMessage) else "assistant"
        ):
            st.markdown(message.content)

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            try:
                response = run_agent(
                    query=prompt, id=st.session_state.current_conversation_id
                )
                st.session_state.messages.append(AIMessage(content=response))

                with st.chat_message("assistant"):
                    st.markdown(response)

            except Exception as e:
                logging.error(
                    f"FILE: app/main.py INFO: Error in Streamlit chat interface. DETAILS: {e}",
                    exc_info=True,
                )
                st.error(f"An error occurred: {e}")

                error_msg = "Sorry, an internal error occurred."
                st.session_state.messages.append(AIMessage(content=error_msg))
                with st.chat_message("assistant"):
                    st.markdown(error_msg)


# showing the system prompt
with st.expander("View System Preamble"):
    st.text(preamble)
