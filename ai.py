import streamlit as st
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from gtts import gTTS
import speech_recognition as sr
import io

# Load environment variables
load_dotenv()

def generate_ai_response(prompt, client, context=""):
    system_message = f"You are an AI Assistant specialized in all knowledge:. Context: {context}"
    
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    response = ""
    for message in client.chat_completion(
        messages=messages,
        max_tokens=1200,
        stream=True
    ):
        response += message.choices[0].delta.content or ""
    
    return response


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


def text_to_speech(text):
    tts = gTTS(text)
    return tts

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        st.write("Processing...")
    try:
        text = recognizer.recognize_google(audio)
        #st.write(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        st.write("Could not understand audio")
    except sr.RequestError as e:
        st.write(f"Error with the Google Speech Recognition service; {e}")
    return ""

def main():
    st.title('🤖 Advisory AI Assistant')

    # Sidebar for settings and file upload
    with st.sidebar:
        st.title('🤖 Advisory Assistant Settings')
        hf_api_token = "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"
        
        st.button('Clear Chat History', on_click=clear_chat_history)
        
        # Upload a text file
        st.subheader("Upload a Text File (200 MB limit)")
        txt_file = st.file_uploader("Choose a text file", type=["txt"], accept_multiple_files=False)
        text_data = ""
        if txt_file:
            if txt_file.size <= 2e8:  # 200 MB limit
                text_data = txt_file.read().decode("utf-8")
                st.write("Text file uploaded successfully!")
            else:
                st.write("File size exceeds the 200 MB limit.")
        
        # Upload a CSV file
        st.subheader("Upload a CSV File (200 MB limit)")
        csv_file = st.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=False)
        df = None
        if csv_file:
            if csv_file.size <= 2e8:  # 200 MB limit
                df = pd.read_csv(csv_file)
                st.write("CSV file uploaded successfully!")
                st.dataframe(df.head())
            else:
                st.write("File size exceeds the 200 MB limit.")


    # Initialize the InferenceClient
    client = InferenceClient(
        token=hf_api_token
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
    # Prepare context from uploaded files
    context = ""
    if text_data:
        context += f"Text data: {text_data[:1000]}... "  # Limiting context size
    if df is not None:
        # Summarize the CSV data
        csv_summary = df.head(5).to_string(index=False)
        context += f"CSV data sample:\n{csv_summary}\n"
    
    # Display chat history with TTS option
    for i, message in enumerate(st.session_state.messages):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if message["role"] == "assistant":
            with col2:
                if st.button("🔊", key=f"tts_{i}"):  # Use the loop index 'i' to ensure a unique key
                    tts = text_to_speech(message["content"])
                    tts.save(f"response_{i}.mp3")  # Save the file with a unique name using 'i'
                    st.audio(f"response_{i}.mp3")

    # User input with STT option
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        user_input = st.chat_input("Type your message here:")
    with col2:
        if st.button("🎤"):
            recognized_text = speech_to_text()
            if recognized_text:
                user_input = recognized_text

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate AI response with context
        response = generate_ai_response(user_input, client, context=context)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()