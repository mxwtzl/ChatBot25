import getpass
import os
import sys
from typing import Any, Dict, List
import json

from langchain_core.output_parsers import StrOutputParser # wandelt Modellantwort standardisiert in String um
from langchain_core.prompts import PromptTemplate # erstellt Prompt-Objekt hauptsächlich um Platzhalter füllbar zu machen
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # ermöglicht strukturierte Aufteilung von Rollen
from langchain_core.outputs import LLMResult # ermöglicht strukturierte Ausgabe von LLMs

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler

# Warnungen unterdrücken
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# API-Key laden
load_dotenv()
API_KEY = os.getenv("CHAT_AI_ACCESS_KEY")
if not API_KEY:
    API_KEY = getpass.getpass("Enter your CHAT_AI_ACCESS_KEY: ")

# Callback zur Protokollierung
class CustomCallback(BaseCallbackHandler):
    def __init__(self):
        self.messages = {}

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        self.messages["on_llm_start_prompts"] = prompts
        self.messages["on_llm_start_kwargs"] = kwargs

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.messages["on_llm_end_response"] = response
        self.messages["on_llm_end_kwargs"] = kwargs

    def raise_error(self, error: Exception) -> None:
        print(f"Callback error: {error}")



class BotAgent:

    STATE_UNCLE = "uncle"
    STATE_HOST = "host"

    def __init__(self):

        # Initialisiere LLMs mit API-Key und Basis-URL
        self.llm = ChatOpenAI(
            model="llama-3.1-sauerkrautlm-70b-instruct",
            temperature=0.7,
            logprobs=True,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )

        # Textklassifizierer für die Unterscheidung zwischen Onkel und Host
        self.text_classifier_llm = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.01,
            logprobs=True,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )

        # Initialisiere den Zustand des Bots -> Host
        self.state = BotAgent.STATE_HOST

        self.host_chain = self.get_host_prompt()
        self.uncle_chain = self.get_uncle_prompt()
        self.text_classifier = self.create_text_classifier()

    def get_host_prompt(self):
        prompt = """
Du bist ein menschlicher, neutraler Spielleiter in einem Rollenspiel in der Rolle des Gastgebers. Der Spieler führt ein schwieriges Gespräch mit einem populistischen Familienmitglied (dem Onkel). Du bist empathisch, reflektiert und hilfsbereit. Du beurteilst nicht direkt, sondern gibst Feedback auf Meta-Ebene.
Beginne das Spiel mit einer freundlichen Begrüßung, indem du den Spieler bittest sich vorzustellen und fragst nach Namen, Geschlecht und wie anstrengend der Onkel sein kann, neben dem du ihn setzt. Anschließend sagst du, dass du dich in der Küche befindest und noch einige Vorbereitungen treffen musst und bei Bedarf (Hilfe, Pause, mehr Informationen) gerufen werden kannst.
Wenn du aktiv wirst, tue Folgendes:
- Fasse die Situation neutral zusammen.
- Gib bei Bedarf emotionale Unterstützung oder Deeskalationstipps.
- Wenn der Spieler darum bittet, gib reflektiertes Feedback zu Argumentationsstil oder benutzten Manipulationstechniken.
- Führe einen verdeckten Score basierend auf Deeskalation, Sachlichkeit und Empathie, welcher zwischen -5 und 5 liegt.
- Wenn der Score 3 Mal in Folge unter 4 fällt, greife kurz ein, um Hilfe zu leisten.
- Gib auf Wunsch Meta-Feedback.

Aktueller Punktestand: {score}/100

Bisherige Unterhaltung:
{conversation_history}

Spieler sagt oder fragt:
{player_input}
"""
        # Umwandlung String -> Prompt
        chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return chain

    def get_uncle_prompt(self):
        prompt = """
Du spielst den Onkel beim Abendessen, der eine stark populistische Meinung vertritt. Du redest viel über "früher war alles besser", bist anti-woke und provozierst gerne. Rede ausschließlich in der Rolle des Onkels.

Bisherige Unterhaltung:
{conversation_history}

Spieler sagt oder fragt:
{player_input}
"""
        # Umwandlung String -> Prompt
        chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return chain

    def create_text_classifier(self):
        prompt = """
Klassifiziere die folgende Eingabe in eine der Kategorien: [uncle, host].

Eingabe:
{player_input}

Antwort:
"""
        # Umwandlung String -> Prompt
        chain = PromptTemplate.from_template(prompt) | self.text_classifier_llm | StrOutputParser()
        return chain 

    def classify_state(self, user_message: str):
        classification_callback = CustomCallback()
        text_classification = self.text_classifier.invoke(
            user_message,
            {"callbacks": [classification_callback], "stop_sequences": ["\n"]},
        )
        
        if text_classification.find("\n") > 0:
            text_classification = text_classification[
                0 : text_classification.find("\n")
            ]
        text_classification = text_classification.strip()

        if text_classification == "uncle":
            self.state = BotAgent.STATE_UNCLE
        elif text_classification == "host":
            self.state = BotAgent.STATE_HOST

        if text_classification in [BotAgent.STATE_UNCLE, BotAgent.STATE_HOST]:
            self.state = text_classification
        return text_classification, classification_callback

    def get_response(self, user_message :str, chat_history: str, score: int = 50):
        text_classification, classification_callback = self.classify_state(user_message)
        chain = self.host_chain if self.state == BotAgent.STATE_HOST else self.uncle_chain

        response_callback = CustomCallback()
        response = chain.invoke(
            {
                "user_message": str(user_message),
                "chat-history": str(chat_history),
                "score": score,
            },
            {"callbacks": [response_callback], "stop_sequences": ["\n"]},
        )

        log_message = {
            "user_message": str(user_message),
            "chatbot_response": str(response),
            "agent_state": self.state,
            "classification": {
                "result": text_classification,
                "llm_details": {
                    key: value
                    for key, value in classification_callback.messages.items()
                },
            },
            "chatbot_response": {
                key: value for key, value in response_callback.messages.items()
            },
        }

        return response, log_message
    
class LogWriter:

    def __init__(self):
        self.conversation_logfile = "conversation.jsonp"
        if os.path.exists(self.conversation_logfile):
            os.remove(self.conversation_logfile)

    # helper function to make sure json encoding the data will work
    def make_json_safe(self, value):
        if type(value) == list:
            return [self.make_json_safe(x) for x in value]
        elif type(value) == dict:
            return {key: self.make_json_safe(value) for key, value in value.items()}
        try:
            json.dumps(value)
            return value
        except TypeError as e:
            return str(value)

    def write(self, log_message):
        with open(self.conversation_logfile, "a") as f:
            f.write(json.dumps(self.make_json_safe(log_message), indent=2))
            f.write("\n")
            f.close()


def test_bot_agent():
    bot = BotAgent()

    # Test 1: Initialer Zustand sollte der Host sein
    assert bot.state == BotAgent.STATE_HOST
    print("Test 1: Initialer Zustand ist Host")

    # Test 2: Input klassifizieren
    input_uncle = "Früher war alles besser und die Flüchtlinge schaden uns!"
    state = bot.classify_state(input_uncle)
    print(f"Test 2: Klassifizierter Zustand: {state}")

    # Test 3: Antwort holen im aktuellen Zustand (kann je nach Klassifikation host/uncle sein)
    conversation_history = "" # ohne vorherigen Gesprächsverlauf
    player_input = "Hallo, ich bin bereit zu spielen."
    response_data = bot.get_response(conversation_history, player_input)
    print("Test 3: Bot antwortet:", response_data["bot_response"])

    # Test 4: Simuliere Onkel-Antwort
    conversation_history = "Spieler: Ich glaube, Migration bringt auch Chancen.\nOnkel: Bla bla bla, früher war alles besser!"
    player_input = "Warum meinst du, dass früher alles besser war?"
    bot.classify_state(player_input)
    response_data = bot.get_response(conversation_history, player_input)
    print("Test 4: Onkel-Antwort:", response_data["bot_response"])


if __name__ == "__main__":
    test_bot_agent()
