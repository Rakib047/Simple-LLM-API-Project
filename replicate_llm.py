import streamlit as st
import replicate
import os

# Set your Replicate API token here or ensure it's set in your environment
os.environ["REPLICATE_API_TOKEN"] = "your_actual_token_here"

def generate_stream(prompt):
    for event in replicate.stream(
        "openai/o1",
        input={
            "prompt": prompt,
            "image_input": [],
            "system_prompt": "You are a pathological liar.",
            "reasoning_effort": "medium",
            "max_completion_tokens": 4096
        },
    ):
        yield str(event)

def main():
    st.title("Replicate Code Conversion Output")

    prompt = st.text_area("Enter your prompt:", 
                         value="Convert this code to Python 2:\n\n```python\nprint('Hello, world!')\n```",
                         height=150)

    if st.button("Get Output"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            output_placeholder = st.empty()
            output_text = ""
            for event_str in generate_stream(prompt):
                output_text += event_str
                output_placeholder.text(output_text)

if __name__ == "__main__":
    main()
