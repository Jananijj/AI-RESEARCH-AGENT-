import streamlit as st
import requests

# FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000"  # Change to your deployed FastAPI URL

# Streamlit app title
st.title("Research Agent")

# User input for the query
query = st.text_input("Enter your research query:")

if query:
    # Send query to FastAPI backend
    response = requests.post(f"{FASTAPI_URL}/", json={"query": query})

    # Check for valid response
    if response.status_code == 200:
        result = response.json()
        if "result" in result:
            st.write("Research Result:")
            st.write(result["result"])
        else:
            st.error("Error: No result found in the response.")
    else:
        st.error(f"Error: Received status code {response.status_code} from FastAPI.")
