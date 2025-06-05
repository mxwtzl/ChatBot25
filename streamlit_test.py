import streamlit as st
st.write("This is a local Streamlit test")

message = st.chat_message("assistant")
message.write("This is the assistant")

# With notation
with st.chat_message("user"):
    st.write("Hello world")

prompt = st.chat_input("Enter your prompt here")
if prompt:
    st.write(f"Your prompt: {prompt}")