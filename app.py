import os
import streamlit as st
from streamlit_chat import message
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import time
from langchain_pinecone.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.fireworks import ChatFireworks

# Streamlit App Configuration
st.set_page_config(page_title="Docu-Help", page_icon="🟩")
st.markdown("<h1 style='text-align: center;'>Ask away:</h1>", unsafe_allow_html=True)

# Read API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINE_API_KEY = os.getenv("PINE_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = 'true'
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = "docu-help"

# Sidebar for model selection and Pinecone index name input
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("gpt-3.5-turbo-1106", "gpt-4-0125-preview", "mixtral"))
pinecone_index_name = st.sidebar.text_input("Enter Pinecone Index Name")

# Initialize session state variables if they don't exist
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "system", "content": "You are a helpful assistant."}]

if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Function to generate a response using App 2's functionality
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    embed = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

    pc = Pinecone(api_key=PINE_API_KEY)
    index = pc.Index(pinecone_index_name)
    time.sleep(1)  # Ensure index is ready
    index.describe_index_stats()

    vectorstore = PineconeVectorStore(index, embed, "text")
    retriever = vectorstore.as_retriever()

    template = """You are an expert software developer who specializes in APIs. Answer the user's question based only on the following context:
{context}
Question: {question}
"""
    prompt_template = ChatPromptTemplate.from_template(template)

    if model_name == "mixtral":
        chat_model = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")
    else:
        stream_params = {
            "temperature": 0,
            "model": model_name,
            "openai_api_key": OPENAI_API_KEY,
            "stream": True,
        }
        chat_model = ChatOpenAI(**stream_params)

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: x["context"]))
        | prompt_template
        | chat_model
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain)

    response = rag_chain_with_source.invoke(prompt)
    answer = response['answer']  # Extracting the 'answer' part
    sources = [doc.metadata['source'] for doc in response['context']]

    formatted_response = f"Answer: {answer}\n\nSources:\n" + "\n".join(sources)

    time.sleep(1)

        # Stream the response character by character
    for char in formatted_response:
        st.write(char, append="")
        time.sleep(0.01)

    st.session_state['messages'].append({"role": "assistant", "content": formatted_response})
    return formatted_response

# Container for chat history and text box
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
