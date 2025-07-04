import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import re
from openai import OpenAI
from functions import chunk_text, get_embedding, input_query, response_use_llm
import numpy as np
import faiss
import tiktoken
import os

load_dotenv()

st.title("Chat with Your Document")
st.subheader('Upload a PDF file and ask questions to explore its content.')

st.session_state.pdf_files = None
st.session_state.pdf_text = ''
st.session_state.index = None
st.session_state.chunks = None
st.session_state.thread = ''


uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    clean_text = re.sub(r'\s+', ' ', text).strip()
    # st.session_state.pdf_text = clean_text
    chunks = chunk_text(clean_text)
    st.session_state.chunks = chunks
    embeddings = [get_embedding(chunk) for chunk in chunks]

    dimension = len(embeddings[0])
    st.session_state.index = faiss.IndexFlatL2(dimension)

    st.session_state.index.add(np.array(embeddings).astype("float32"))
    st.success("PDF content loaded!")






# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask some questions"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    extract_chunk_idx = input_query(st.session_state.index,st.session_state.chunks, prompt )

    st.session_state.thread += ('user: ' + prompt +'\n\n')

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    response = response_use_llm(st.session_state.chunks, extract_chunk_idx ,prompt, st.session_state.thread)

    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history

    st.session_state.thread += ('assistant: ' + response +'\n\n')

    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("""
---
🖋️ *Developed by Mark Samuel Nicasio (Sam)*
[GitHub](https://github.com/yourusername)
""")