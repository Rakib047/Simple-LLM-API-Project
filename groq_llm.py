import streamlit as st
import os
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def main():
    # Load API key from environment variable for security
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        st.error("GROQ_API_KEY not set in environment variables!")
        return

    st.title('Code Converter AI')
    st.write('I’m your friendly Chatbot who can convert your code to different versions and languages!')

    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        [
            'llama-3.3-70b-versatile',
            'llama-3.1-8b-instant',
            'llama3-70b-8192',
            'llama3-8b-8192',
            'gemma2-9b-it',
            'meta-llama/Llama-Guard-4-12B',
            'deepseek-r1-distill-llama-70b',
        ]
    )

    conversational_memory_length = st.sidebar.slider('Conversational memory length', 1, 10, value=5)

    # New additions for generation parameters
    temperature = st.sidebar.slider('Temperature', 0.0, 1.0, value=0.7, step=0.05)
    top_p = st.sidebar.slider('Top-p (nucleus sampling)', 0.0, 1.0, value=0.9, step=0.05)
    max_tokens = st.sidebar.slider('Max completion tokens', 64, 2048, value=512, step=64)

    # Initialize memory and conversation chain once
    if 'memory' not in st.session_state or st.session_state.memory_k != conversational_memory_length:
        st.session_state.memory = ConversationBufferWindowMemory(k=conversational_memory_length)
        st.session_state.memory_k = conversational_memory_length
        st.session_state.chat_history = []

    # Rehydrate memory from chat history
    for message in st.session_state.chat_history:
        st.session_state.memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )

    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens
    )

    conversation = ConversationChain(
        llm=groq_chat,
        memory=st.session_state.memory
    )

    user_question = st.text_input("Ask a question:")

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.experimental_rerun()

    if user_question:
        with st.spinner('Generating response...'):
            try:
                response = conversation(user_question)
                message = {'human': user_question, 'AI': response['response']}
                st.session_state.chat_history.append(message)
            except Exception as e:
                st.error(f"Error from Groq API: {e}")
                return

    # Display chat history with simple formatting
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['human']}")
        st.markdown(f"**Bot:** {chat['AI']}")

if __name__ == "__main__":
    main()
