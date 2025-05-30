import streamlit as st
import google.generativeai as genai
import os

# Configure the API key (make sure GOOGLE_API_KEY is set in your environment)
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# List of available models you want to offer
MODELS = [
    "gemini-1.5-flash-001",
    "gemini-1.5-extended-001",
    "gemini-2.0-flash",
    "gemini-code",
    "whisper-large-v3",
    "whisper-large-v3-turbo"
]

def generate_response(model_name: str, generation_config: dict, prompt: str) -> str:
    model = genai.GenerativeModel(model_name, generation_config=generation_config)
    response = model.generate_content([prompt])
    return response.text

def main():
    st.title("Gemini LLM - Code Version Converter Chatbot")
    st.write("Enter your code conversion question or prompt below:")

    # Sidebar controls
    st.sidebar.header("Model & Generation Config")
    model_name = st.sidebar.selectbox("Choose Model", MODELS)

    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.9, 0.05)
    top_p = st.sidebar.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.95, 0.05)
    max_tokens = st.sidebar.slider("Max output tokens", 64, 2048, 1024, 64)

    prompt = st.text_area("Your prompt:", height=150)

    if st.button("Generate Response"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_tokens
            }
            with st.spinner("Generating..."):
                answer = generate_response(model_name, generation_config, prompt)
                st.markdown("### Response:")
                st.write(answer)

if __name__ == "__main__":
    main()
