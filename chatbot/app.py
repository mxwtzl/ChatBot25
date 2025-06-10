import streamlit as st
import requests
import json

# Konfiguration der Seite
st.set_page_config(
    page_title="Uncle Bot",
    page_icon="🤖",
    layout="centered"
)

# CSS für besseres Styling
st.markdown("""
<style>
    /* Streamlit UI Elemente ausblenden */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.bot {
        background-color: #f0f2f6;
    }
    .chat-message .avatar {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .input-container {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 1rem 0;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

# Initialisierung des Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_state" not in st.session_state:
    st.session_state.current_state = "onkel"
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Titel und Beschreibung
st.title("Uncle Bot")
st.markdown("""
Welcome to the first version of the Uncle Bot!
""")

# Status-Anzeige
NEUTRAL_AVATAR = "👩‍🦱"
ONKEL_AVATAR = "👨‍🦳"
state_emoji = "👩‍🦱" if st.session_state.current_state == "neutral" else "👨‍🦳"
st.markdown(f"**You are talking to:** {state_emoji}")

# Chat-Verlauf anzeigen
# TO-DO: Anpassung damit state_emoji sich bei Änderung nicht für gesamten Verlauf anpasst
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div>👤 <b>Du:</b></div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if message["role"] == "neutral":
                st.markdown(f"""
                <div class="chat-message neutral">
                        <div>{NEUTRAL_AVATAR} <b>Host:</b></div>
                        <div>{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message onkel">
                    <div>{ONKEL_AVATAR} <b>Uncle:</b></div>
                    <div>{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

# Eingabefeld in einem Container
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    user_input = st.text_input("Enter your message:", key=f"user_input_{st.session_state.input_key}")
    st.markdown('</div>', unsafe_allow_html=True)

if user_input and user_input != st.session_state.last_input:
    # Nachricht zum Chat-Verlauf hinzufügen
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.last_input = user_input
    
    # API-Anfrage senden
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "message": user_input,
                "chat_history": [msg["content"] for msg in st.session_state.messages]
            }
        )
        response_data = response.json()
        st.session_state.current_state = response_data["state"]
        # Bot Antwort zum Chat-Verlauf hinzufügen
        # aktuelles Problem: Session-State hängt bei Anzeige immer hinter tatsächlichem Session-State zurück (erst bei zweiter Eingabe korrekt)
        if st.session_state.current_state == "onkel":
            st.session_state.messages.append({"role": "onkel", "content": response_data["response"]})
        else:
            st.session_state.messages.append({"role": "neutral", "content": response_data["response"]})
        
        # st.session_state.messages.append({"role": "neutral", "content": response_data["response"]})
        # st.session_state.current_state = response_data["state"]
        
        # Eingabefeld leeren durch Erhöhung des Keys
        st.session_state.input_key += 1
        st.experimental_rerun()
        
    except Exception as e:
        st.error(f"Fehler bei der Kommunikation mit dem Server: {str(e)}") 