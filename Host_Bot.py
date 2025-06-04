import getpass #Für sichere Passworteingabe
import os
import sys
from typing import Any, Dict, List
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler

# Warnungen unterdrücken
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# API
load_dotenv()
API_KEY = os.getenv("CHAT_AI_ACCESS_KEY")
if not API_KEY:
    API_KEY = getpass.getpass("Enter your CHAT_AI_ACCESS_KEY: ")

# Callback zur Protokollierung
class CustomCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.messages = {}

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self.messages["on_llm_start_prompts"] = prompts
        self.messages["on_llm_start_kwargs"] = kwargs # weitere Infos wie Modellname, Einstellungen ...

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.messages["on_llm_end_response"] = response
        self.messages["on_llm_end_kwargs"] = kwargs

    def raise_error(self, error: Exception) -> None:
        print(f"Callback error: {error}")



class BotAgent:
    STATE_UNCLE = "uncle"
    STATE_HOST = "host"

    def __init__(self):
        self.llm = ChatOpenAI(
            model="llama-3.1-sauerkrautlm-70b-instruct",
            temperature=0.7,
            logprobs=True,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )

        self.text_classifier_llm = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.01,
            logprobs=True,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )

        self.state = BotAgent.STATE_HOST

        self.host_chain = self.create_prompt_chain(self.get_host_prompt())
        self.uncle_chain = self.create_prompt_chain(self.get_uncle_prompt())
        self.text_classifier = self.create_text_classifier()

    def get_host_prompt(self):
        return """
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

    def get_uncle_prompt(self):
        return """
Du spielst den Onkel beim Abendessen, der eine stark populistische Meinung vertritt. Du redest viel über "früher war alles besser", bist anti-woke und provozierst gerne. Rede ausschließlich in der Rolle des Onkels.

Bisherige Unterhaltung:
{conversation_history}

Spieler sagt oder fragt:
{player_input}
"""

    def create_prompt_chain(self, prompt_template: str):
        prompt = PromptTemplate.from_template(prompt_template)
        return prompt | self.llm | RunnableLambda(lambda x: x.content if hasattr(x, "content") else str(x))

    def create_text_classifier(self):
        prompt_template = PromptTemplate.from_template(
            """
Klassifiziere die folgende Eingabe in eine der Kategorien: [uncle, host].

Eingabe:
{player_input}

Antwort:
"""
        )
        return prompt_template | self.text_classifier_llm | RunnableLambda(lambda x: x.content.strip().lower())

    def classify_state(self, player_input: str):
        result = self.text_classifier.invoke({"player_input": player_input})
        if result in [BotAgent.STATE_UNCLE, BotAgent.STATE_HOST]:
            self.state = result
        return self.state

    def get_response(self, conversation_history: str, player_input: str, score: int = 50):
        chain = self.host_chain if self.state == BotAgent.STATE_HOST else self.uncle_chain
        response_callback = CustomCallback()

        response = chain.invoke(
            {
                "conversation_history": conversation_history,
                "player_input": player_input,
                "score": score,
            },
            config={"callbacks": [response_callback]},
        )

        return {
            "bot_response": response,
            "log": response_callback.messages,
            "state": self.state
        }


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
