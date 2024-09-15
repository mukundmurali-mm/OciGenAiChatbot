from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms.oci_generative_ai import OCIGenAI
import streamlit as st

MESSAGES_ = """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """

st.set_page_config(page_title="OCI GenAI chatbot", page_icon="https://oracle.gallerycdn.vsassets.io/extensions/oracle/oci-core/1.0.5/1715101068442/Microsoft.VisualStudio.Services.Icons.Default")
st.title("OCI GenAI chatbot")

"""
A basic example of using OCI Gen AI services and Langchain wrapped around streamlit to provide a simple chatbot. 
View the
[source code for this app](https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py).
"""

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# The Authentication for this code is taken from your local config located in ~/.oci/config file and default profile is used. Make sure your API key is configured in the config accordingly.

llm = OCIGenAI(
    model_id="cohere.command", # Update the OCI GenAI model ID here
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaajbsk4btzn6oymimjelztkawcgf2qwq3jlnhrwn4pnnla2umcsmva",# Update the OCI GenAI compartment ID here
    is_stream=True,
    model_kwargs={"temperature": 0, "max_tokens": 500} # Update the OCI GenAI model parameters here
)

# Set up the LangChain, passing in Message History
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an OCI AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.stream({"question": prompt}, config)
    st.chat_message("ai").write(response)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    view_messages.json(st.session_state.langchain_messages)
