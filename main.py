import os
import streamlit as st
from bs4 import BeautifulSoup
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
import chromadb
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
    st.markdown("""
    This chatbot extracts text from a specified website in real time and answers questions about the content provided.
    You can ask questions related to the website content and get accurate responses based on the extracted data.\n
    For example, you might ask questions like ***"What is the main topic of this page?"*** or,\n
    ***"Can you summarize the key points?"***.\n
    The project repository can be found [on my Github](https://github.com/muhammad-ahsan12/MakTek-internship-Task.git).
    """)
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')

    key = st.text_input("Enter the Google API key", type='password')
    groq_api = st.text_input("Enter the Groq API key", type='password')
    url = st.text_input("Insert the website URL")
    user_question = st.text_input("Ask a question (query/prompt)")

    if st.button("Submit Query", type="primary"):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")
        
        os.environ['GOOGLE_API_KEY'] = key  # Set the Google API key from user input
        
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

        # Create a Chroma vector database from the text documents
        vectordb = Chroma.from_texts(texts=docs, embedding=embeddings)
        vectordb.persist()

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Use a ChatGroq model for question-answering
        llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api)

        # Create a ConversationalRetrievalChain instance from the model and retriever
        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            verbose=True,
        )

        chat_history = []

        # Run the prompt and return the response
        response = chain({"question": user_question, "chat_history": chat_history})

        # Extract and display the result
        if 'answer' in response:
            st.write(response['answer'])
        else:
            st.write("Answer not found.")

if __name__ == '__main__':
    main()
