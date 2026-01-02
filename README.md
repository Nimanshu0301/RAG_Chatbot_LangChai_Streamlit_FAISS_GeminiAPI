# RAG_Chatbot_LangChai_Streamlit_FAISS_GeminiAPI

## 📚 Chat with Multiple PDFs using LangChain & Google Gemini

This project is a Streamlit-based web application that allows users to upload multiple PDF documents and interact with them using natural language questions. It leverages LangChain, Google Gemini (Generative AI), and FAISS vector search to provide accurate, context-aware answers extracted directly from the uploaded PDFs.

## 🚀 Features

Upload multiple PDF files simultaneously

Automatically extract and chunk text from PDFs

Generate vector embeddings using Google Generative AI

Store and search embeddings using FAISS

Ask questions and receive context-grounded answers

Maintains conversation history with timestamps and source PDFs

Download chat history as a CSV file

Clean, chat-style UI built with Streamlit

## 🛠️ Tech Stack

Frontend / UI: Streamlit

LLM & Embeddings: Google Gemini (gemini-1.5-flash)

Framework: LangChain

Vector Database: FAISS

PDF Processing: PyPDF2

Language: Python


## Create Virtual Environment:

python3 -m venv myenv

source myenv/bin/activate
