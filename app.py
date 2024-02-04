import os
import streamlit as st
from streamlit_chat import message
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import time
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

# Streamlit App Configuration
st.set_page_config(page_title="Docu-Help", page_icon="🟩")
st.markdown("<h1 style='text-align: center;'>Ask any question about a specified service:</h1>", unsafe_allow_html=True)

# Read API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINE_API_KEY = os.getenv("PINE_API_KEY")

# Sidebar for model selection and Pinecone index name input
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("gpt-3.5-turbo-1106", "gpt-4-0125-preview"))
pinecone_index_name = st.sidebar.text_input("Enter Pinecone Index Name")

# Initialize session state variables if they don't exist
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Function to generate a response using App 2's functionality
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    # Your existing setup with user inputs from App 2
    embed = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

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
    chat_model = ChatOpenAI(temperature=0, model=model_name, openai_api_key=OPENAI_API_KEY)

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
    st.session_state['messages'].append({"role": "assistant", "content": response})
    return response

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

