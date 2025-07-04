import streamlit as st
from PyPDF2 import PdfReader

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("My Streamlit Chat App with PDF Upload 📚💬")

# Input text box
user_input = st.text_input("You:", "")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Display uploaded PDF content
if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    st.text_area("PDF Content:", text, height=200)

# Handle user input
if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate a simple response (mock)
    response = f"Echo: {user_input}"
    st.session_state.messages.append({"role": "bot", "content": response})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")
