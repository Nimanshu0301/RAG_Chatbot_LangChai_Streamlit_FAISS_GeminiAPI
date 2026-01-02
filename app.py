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
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

#embedding this chunks and storing in vector db 
def get_vectorstore(text_chunks, model_name, api_key=None):
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="mode1/embedding-001", google_api_key=api_key)
        
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