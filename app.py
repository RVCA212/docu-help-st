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

# Streamlit App
def main():
    st.title("Chat with Documentation.com")

    # API keys (Read from environment variables)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINE_API_KEY = os.getenv("PINE_API_KEY")

    # Sidebar for model selection and Pinecone index name input
    with st.sidebar:
        st.header("Configuration")
        model_option = st.radio(
            "Choose your model:",
            ("gpt-4-1106-preview", "gpt-3.5-turbo-1106")
        )
        pinecone_index_name1 = st.text_input("Enter Pinecone Index Name 1")
        pinecone_index_name2 = st.text_input("Enter Pinecone Index Name 2")

    if pinecone_index_name1 and pinecone_index_name2:
        # Your existing setup with user inputs
        model_name = 'text-embedding-ada-002'
        embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

        pc = Pinecone(api_key=PINE_API_KEY)
        index1 = pc.Index(pinecone_index_name1)
        index2 = pc.Index(pinecone_index_name2)
        time.sleep(1)
        index1.describe_index_stats()
        index2.describe_index_stats()

        text_field = "text"
        vectorstore1 = PineconeVectorStore(index1, embed, text_field)
        vectorstore2 = PineconeVectorStore(index2, embed, text_field)
        retriever1 = vectorstore1.as_retriever()
        retriever2 = vectorstore2.as_retriever()

        template = """You are an expert software developer who specializes in APIs. Answer the user's question based only on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(temperature=0, model=model_option, openai_api_key=OPENAI_API_KEY)

        def format_docs_with_sources(docs):
            # This function formats the documents and includes their sources.
            formatted_docs = []
            for doc in docs:
                content = doc.page_content
                source = doc.metadata.get('source', 'Unknown source')
                formatted_docs.append(f"{content}\nSource: {source}")
            return "\n\n".join(formatted_docs)

        def combine_contexts(contexts):
            # This function combines contexts from two retrievers.
            combined_context = "\n\n".join([format_docs_with_sources(ctx) for ctx in contexts])
            return combined_context

        rag_chain = (
            RunnablePassthrough.assign(context=(lambda x: combine_contexts(x["context"])))
            | prompt
            | model
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": (retriever1, retriever2), "question": RunnablePassthrough()}
        ).assign(answer=rag_chain)

        # Query input area
        query = st.text_area("Enter your query:", height=150)
        submit_button = st.button('Submit Query')

        if submit_button:
            with st.spinner('Processing...'):
                # Log the input query to the terminal
                print(f"Input Query: {query}")

                response = rag_chain_with_source.invoke(query)

                # Log the response to the terminal
                print(f"Output Response: {response}")

            st.write(response)

if __name__ == "__main__":
    main()

