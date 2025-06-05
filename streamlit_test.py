import streamlit as st


#st.write("This is a local Streamlit test")

#message = st.chat_message("assistant")
#message.write("This is the assistant")

# With notation
#with st.chat_message("user"):
#    st.write("Hello world")

# basic chat input
#prompt = st.chat_input("Enter your prompt here")
#if prompt:
#    st.write(f"Your prompt: {prompt}")


st.title("Echo Bot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    message = st.chat_message(message["role"]).markdown(message["content"])

    
if prompt := st.chat_input("Your input"):
    message = st.chat_message("user")
    message.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = prompt
    message = st.chat_message("assistant")
    message.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
