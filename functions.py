import tiktoken
from openai import OpenAI
import numpy as np
import os
import streamlit as st

def chunk_text(text, chunk_size=500, overlap=50):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embedding(text):

    # client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
    client = OpenAI(api_key = st.secrets["OPENAI_API_KEY"])
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def input_query(index,chunks,query):
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding]).astype("float32"), k=5)

    return I[0]

def use_llm(prompt):
    # client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
    client = OpenAI(api_key = st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def response_use_llm(chunks, result_idx,query, thread):
    context = "\n\n".join([chunks[idx] for idx in result_idx])

    prompt = f"""
    Answer the question based on the context below.

    Context: ```{context}```

    Previous thread: ```{thread}````

    Question: ```{query}```

    Include some bullet points or table when necessary to make it pleasing to the reader.

    Add follow up response suggestion to the user that they may ask to make the convesation flowing, and make it short and friendly.
    """

    response = use_llm(prompt)

    return response
