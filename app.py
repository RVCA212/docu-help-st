import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import time
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

# Function to get active Pinecone index names
def get_active_pinecone_indexes(api_key):
    pc = Pinecone(api_key=api_key)
    indexes_info = pc.list_indexes()
    # Assuming indexes_info is a dictionary containing a list under the key 'indexes'
    # and each item in that list is a dictionary with keys 'name' and 'status'
    active_index_names = [index['name'] for index in indexes_info['indexes'] if index['status'] == 'READY']
    return active_index_names


# Streamlit App
def main():
    st.title("Chat with Documentation.com")

    # API keys (Read from environment variables)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINE_API_KEY = os.getenv("PINE_API_KEY")

    # Sidebar for Pinecone index selection
    with st.sidebar:
        st.header("Configuration")
        active_indexes = get_active_pinecone_indexes(PINE_API_KEY)
        selected_index_name = st.selectbox("Select Pinecone Index", active_indexes)

    if selected_index_name:
        # Your existing setup with user inputs
        model_name = 'text-embedding-ada-002'
        embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

        pc = Pinecone(api_key=PINE_API_KEY)
        index = pc.Index(selected_index_name)
        time.sleep(1)  # Wait a bit for the index connection
        index.describe_index_stats()

        text_field = "text"
        # Pass the embed object directly instead of embed.embed_query
        vectorstore = PineconeVectorStore(index, embed, text_field)
        retriever = vectorstore.as_retriever()

        template = """You are an expert software developer who specializes in APIs. Answer the user's question based only on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", openai_api_key=OPENAI_API_KEY)
        chain = (RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
                 | prompt
                 | model
                 | StrOutputParser())

        # Query input area
        query = st.text_area("Enter your query:", height=150)  # Adjust the height as needed
        submit_button = st.button('Submit Query')

        if submit_button:
            with st.spinner('Processing...'):
                # Log the input query to the terminal
                print(f"Input Query: {query}")

                response = chain.invoke(query)

                # Log the response to the terminal
                print(f"Output Response: {response}")

            st.write(response)

if __name__ == "__main__":
    main()

