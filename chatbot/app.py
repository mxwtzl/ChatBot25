import streamlit as st
import requests
import json
import re

# FastAPI endpoint URLs
# TO-DO: Klären ob BASE_URL auch bei Deployment auf Server so funktioniert 
BASE_URL = "http://localhost:8000"
SET_USERID_URL = f"{BASE_URL}/set-userid"
SET_LANGUAGE_URL = f"{BASE_URL}/set-language"
CHAT_URL = f"{BASE_URL}/chat"

# Konfiguration der Seite
st.set_page_config(
    page_title="Uncle Bot",
    page_icon="🤖",
    layout="centered"
)

# CSS für Styling
st.markdown("""
<style>
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
    .chat-message.bot, .chat-message.neutral, .chat-message.onkel {
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

# UserID Eingabe validieren
# TO-DO: Eingabevalidierung auf unseren Use Case anpassen
def validate_userid(userid: str) -> bool:
    pattern = r"^[a-zA-Z0-9_]{10}$"
    return bool(re.match(pattern, userid))

# Session-State intitalisieren
if "step" not in st.session_state:
    st.session_state.step = "enter_userid"
if "userid" not in st.session_state:
    st.session_state.userid = None
if "language" not in st.session_state:
    st.session_state.language = "de"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_state" not in st.session_state:
    st.session_state.current_state = "onkel"
if "round_count" not in st.session_state:
    st.session_state.round_count = 0
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Titel und Beschreibung 
# TO-DO: Anpassung damit Titel und Beschreibung auf Deutsch ODER Englisch angezeigt werden können
st.title("Uncle Bot / Onkel Bot")
st.markdown("Welcome to the first version of the Uncle Bot! / Willkomen zur ersten Version des Onkel Bot!")

# Eingabe der UserID
if st.session_state.step == "enter_userid":
    st.subheader("Enter User-ID / Eingabe User-ID")
    userid_input = st.text_input(
        "Pleaser Enter the User-ID provided in the PDF-File / Bitte gib die User-ID aus der PDF-Datei ein",
        placeholder="Enter here / Hier eingeben"
    )
    if st.button("Submit / Bestätigen"):
        if validate_userid(userid_input):
            try:
                response = requests.post(
                    SET_USERID_URL,
                    json={"userid": userid_input}
                )
                response.raise_for_status()
                st.session_state.userid = userid_input
                st.session_state.step = "select_language"
                st.success(response.json().get("message"))
                st.rerun()
            except requests.RequestException as e:
                st.error(f"Error setting user ID: {str(e)}")
        else:
            st.error("Invalid user ID.")
    
# Sprachauswahl
# TO-DO: Sprachauswahl VOR UserID um Anzeigen auf Deutsch ODER Englisch ermöglichen 
elif st.session_state.step == "select_language":
    st.subheader("Select Language / Sprachauswahl")
    language = st.selectbox("Choose language / Wähle Sprache:", ["Deutsch (de)", "English (en)"])
    language_code = "de" if language.startswith("Deutsch") else "en"
    if st.button("Submit Language"):
        try:
            response = requests.post(
                SET_LANGUAGE_URL,
                json={"userid": st.session_state.userid, "language": language_code}
            )
            response.raise_for_status()
            st.session_state.language = language_code
            st.session_state.step = "chat"
            st.success(response.json().get("message"))
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Error setting language: {str(e)}")

# Chat-Logik 
elif st.session_state.step == "chat":
    # Status display
    NEUTRAL_AVATAR = "👩‍🦱"
    ONKEL_AVATAR = "👨‍🦳"
    state_emoji = NEUTRAL_AVATAR if st.session_state.current_state == "neutral" else ONKEL_AVATAR
    persona = "Narrator" if st.session_state.current_state == "neutral" else "Uncle"
    if st.session_state.language == "de":
        persona = "Erzähler" if st.session_state.current_state == "neutral" else "Onkel Gerhard"
        st.markdown(f"**Du sprichst mit:** {state_emoji} {persona} (Round: {st.session_state.round_count}/10)")    
    else:
        st.markdown(f"**You are talking to:** {state_emoji} {persona} (Round: {st.session_state.round_count}/10)")

    # Chatverlauf ausgeben 
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
                role_class = message["role"]
                avatar = NEUTRAL_AVATAR if message["role"] == "neutral" else ONKEL_AVATAR
                role_name = "Host" if message["role"] == "neutral" else "Uncle"
                if st.session_state.language == "de":
                    role_name = "Erzähler" if message["role"] == "neutral" else "Onkel Gerhard"
                st.markdown(f"""
                <div class="chat-message {role_class}">
                    <div>{avatar} <b>{role_name}:</b></div>
                    <div>{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

    # Eingaben im Chat
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        if st.session_state.language == "de":
            user_input = st.text_input(
                "Deine Nachricht:",
                key=f"user_input_{st.session_state.input_key}",
                placeholder="Gib deine Nachricht ein..."
            )
        else:
            user_input = st.text_input(
                "Enter your message:",
                key=f"user_input_{st.session_state.input_key}",
                placeholder="Type your message here..."
            )
        st.markdown('</div>', unsafe_allow_html=True)

    if user_input and user_input != st.session_state.last_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.last_input = user_input

        try:
            response = requests.post(
                CHAT_URL,
                json={
                    "message": user_input,
                    "chat_history": [msg["content"] for msg in st.session_state.messages],
                    "userid": st.session_state.userid,
                    "language": st.session_state.language
                }
            )
            response.raise_for_status()
            response_data = response.json()

            st.session_state.current_state = response_data["state"]
            st.session_state.round_count = response_data["round_count"]

            role = "neutral" if response_data["state"] == "neutral" else "onkel"
            st.session_state.messages.append({
                "role": role,
                "content": response_data["response"]
            })

            if response_data["round_count"] >= 10 and response_data["state"] == "onkel":
                st.warning("Conversation ended after 10 rounds. Start a new session.")
                st.session_state.step = "enter_userid"
                st.session_state.userid = None
                st.session_state.language = "de"
                st.session_state.messages = []
                st.session_state.current_state = "onkel"
                st.session_state.round_count = 0
                st.session_state.input_key += 1
                st.rerun()

            st.session_state.input_key += 1
            st.rerun()

        except requests.RequestException as e:
            st.error(f"Fehler bei der Kommunikation mit dem Server: {str(e)}")