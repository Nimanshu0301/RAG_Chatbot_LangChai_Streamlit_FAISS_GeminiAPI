#import packages

import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime

#imports for langchain

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from datetime import datetime

#to get text chunks from pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#to get text chunks from text
def get_text_chunks(text, model_name):
    if model_name == "Google AI":
        
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

#embedding this chunks and storing in vector db 
def get_vectorstore(text_chunks, model_name, api_key=None):
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# create a conversation chain using langchain
def  get_conversation_chain(model_name, vectorstore=None, api_key=None):
    if model_name == "Google AI":
       
        prompt_template = """Answer the question as detailed as possible from the provided context, 
        make sure to provide all the details with proper structure, if the answer is not in the provided context just say, 
        "answer is not available in the context", don't provide wrong answer.
        Context: {context}
        Question: {question}
        Answer in a concise manner."""

        model = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        chain = load_qa_chain(model, chain_type="stuff", prompt=PROMPT)
        return chain
    
#take user input
def user_input(user_question, model_name, api_key, pdf_docs, Conversation_history):
        if api_key is None or pdf_docs is None:
            st.warning("Please provide the API key and upload PDF documents.")
            return
        
        text_chunks = get_text_chunks(get_pdf_text(pdf_docs), model_name)
        vectorstore = get_vectorstore(text_chunks, model_name, api_key)
        user_question_output = ""
        response_output = ""
        if model_name == "Google AI":
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversation_chain("Google AI", vectorstore=new_db, api_key=api_key)
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            user_question_output = user_question
            response_output = response['output_text']                          
            pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
            Conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ", ".join(pdf_names)))

            st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
            .chat-message .info {{
                font-size: 0.8rem;
                margin-top: 0.5rem;
                color: #ccc;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
            </div>
            <div class="message">{response_output}</div>
            </div>
            
        """,
        unsafe_allow_html=True
    )

# main function

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.header("📚 Chat with Multiple PDFs")

    # ✅ SESSION STATE INITIALIZATION
    if "Conversation_history" not in st.session_state:
        st.session_state.Conversation_history = []

    # ================= SIDEBAR =================
    with st.sidebar:
        st.title("🔧 Settings")

        api_key = st.text_input("Google API Key", type="password")
        st.markdown("[Get API Key](https://ai.google.dev/)")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Reset Conversation"):
            st.session_state.Conversation_history = []
            st.experimental_rerun()

    # ================= CHAT =================
    user_question = st.text_input("Ask a question from the PDFs")

    # if user_question:
    #     user_input(user_question, api_key, pdf_docs)
    if user_question:
        user_input(
            user_question=user_question,
            model_name="Google AI",
            api_key=api_key,
            pdf_docs=pdf_docs,
            Conversation_history=st.session_state.Conversation_history
    )

    # ================= DOWNLOAD HISTORY =================
    if st.session_state.Conversation_history:
        df = pd.DataFrame(
            st.session_state.Conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp", "PDFs"]
        )

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()

        st.sidebar.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="Conversation_history.csv">'
            f'<button>⬇️ Download History</button></a>',
            unsafe_allow_html=True
        )

    st.snow()

# ===================== RUN =====================

if __name__ == "__main__":
    main()
