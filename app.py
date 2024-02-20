import os
import streamlit as st
from streamlit_chat import message
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import time
import queue
import asyncio
from langchain_pinecone.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.fireworks import ChatFireworks

# Streamlit App Configuration
st.set_page_config(page_title="Docu-Help", page_icon="🟩")
st.markdown("<h1 style='text-align: center;'>ask away:</h1>", unsafe_allow_html=True)

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
# Initialize queue and session state variables if they don't exist
if 'generated' not in st.session_state:
    st.session_state['generated'] = queue.Queue()
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Function to generate a response using App 2's functionality
# Adjusted generate_response_stream function
async def generate_response_stream(prompt, queue):
    output = await generate_response(prompt)
    return output
    st.session_state['messages'].append({"role": "user", "content": prompt})
    # Your existing setup with user inputs from App 2
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
    
    # Assuming `response` is the object containing 'context', 'question', and 'answer' as shown
    answer = response['answer']  # Extracting the 'answer' part
    
    # Extracting sources from the 'context'
    sources = [doc.metadata['source'] for doc in response['context']]
    
    formatted_response = f"Answer: {answer}\n\nSources:\n" + "\n".join(sources)

    # Add the formatted_response to the queue
    queue.put(formatted_response)
    
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
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(None)  # Initialize as None

            # Start the async function to stream the output
            output_generator = asyncio.create_task(generate_response_stream(user_input))

            # Update the output every 0.1 seconds
            while output_generator.done() is False:
                st.session_state['generated'][-1] = output_generator.result()  # Update the generated message
                time.sleep(0.1)

if not st.session_state['generated'].empty():
    with response_container:
        while not st.session_state['generated'].empty():
            msg = st.session_state['generated'].get()
            message(msg, key=str(len(st.session_state['generated']) - 1))

