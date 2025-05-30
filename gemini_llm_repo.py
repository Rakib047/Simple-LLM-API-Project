import streamlit as st
import google.generativeai as genai
import os
import requests

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

MODELS = [
    "gemini-1.5-flash-001",
    "gemini-code",
    "gemini-1.5-extended-001",
    "gemini-2.0-flash",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    # Add more models as needed
    # Add more if needed
]

def fetch_code_from_github(raw_url: str) -> str:
    try:
        response = requests.get(raw_url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Failed to fetch code: {e}")
        return ""

def generate_response(model_name: str, generation_config: dict, prompt: str) -> str:
    model = genai.GenerativeModel(model_name, generation_config=generation_config)
    response = model.generate_content([prompt])
    return response.text

def main():
    st.title("Gemini Code Version/Language Converter from GitHub File")
    
    # Sidebar configs
    st.sidebar.header("Model & Generation Config")
    model_name = st.sidebar.selectbox("Choose Model", MODELS)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
    max_tokens = st.sidebar.slider("Max output tokens", 64, 2048, 1024, 64)

    st.write("### Step 1: Provide a GitHub raw file URL of the code")
    github_url = st.text_input("GitHub Raw File URL")

    st.write("### Step 2: Choose the target language/version")
    target_language = st.text_input("Target language or version (e.g., Python 3, JavaScript, C++)")

    if st.button("Convert Code"):
        if not github_url.strip():
            st.warning("Please enter a GitHub raw file URL.")
            return
        if not target_language.strip():
            st.warning("Please enter a target language or version.")
            return
        
        code_content = fetch_code_from_github(github_url)
        if not code_content:
            return
        
        prompt = f"Convert the following code to {target_language}:\n\n{code_content}"
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens
        }
        with st.spinner("Converting code..."):
            converted_code = generate_response(model_name, generation_config, prompt)
            st.write("### Converted Code:")
            st.code(converted_code, language=target_language.lower())

if __name__ == "__main__":
    main()
