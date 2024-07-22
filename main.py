import os
import streamlit as st
from langchain.chains import RetrievalQA
from bs4 import BeautifulSoup
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
import requests

# Define the system template for answering questions
system_template = """Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer."""

# Create message templates for system and human messages
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

# Create a chat prompt template from the messages
prompt = ChatPromptTemplate.from_messages(messages)

def main():
    # Set up the Streamlit app interface
    st.title('🦜🔗 Chat With Website')
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')
    key = st.text_input("Enter the Google API key", type='password')
    url = st.text_input("Insert The website URL")
    user_question = st.text_input("Ask a question (query/prompt)")

    if st.button("Submit Query", type="primary"):
        os.environ['GOOGLE_API_KEY'] = key  # Set the Google API key
        
        # Load HTML content from the URL
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        
        # Extract text from the HTML content
        text = soup.get_text(separator='\n')
        
        # Split the text data into chunks
        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=512, chunk_overlap=100)
        docs = text_splitter.split_text(text)

        # Create Google Generative AI embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create a FAISS vector database from the text documents
        vectordb = FAISS.from_texts(texts=docs, embedding=embeddings)

        # Create a retriever from the FAISS vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Use a ChatGroq model for question-answering
        llm = ChatGroq(model="llama3-70b-8192", groq_api_key="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")

        # Create a RetrievalQA instance from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # Run the user's question through the RetrievalQA and display the response
        response = qa.invoke({"query": user_question})
        if 'result' in response:
            st.write(response['result'])
        else:
            st.write("Answer not found.")

if __name__ == '__main__':
    main()
