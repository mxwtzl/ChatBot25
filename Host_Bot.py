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

    STATE_ONKEL = "uncle"
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
        self.host_intro_chain = self.get_host_intro_prompt()
        self.uncle_chain = self.get_uncle_prompt()
        self.text_classifier = self.create_text_classifier()

    def get_host_prompt(self):
        prompt = ("""
            **Rollenbeschreibung: Gastgeberin Alexa**

            Du bist Alexa - eine neutrale, höfliche Gastgeberin eines Tischgesprächs. Du hast alle eingeladen und willst ein respektvolles Gesprächsklima schaffen. Du hörst zu und meldest dich **nur**, wenn du direkt angesprochen wirst.

            **Deine Aufgaben bei Ansprache:**
            - Fasse den bisherigen Gesprächsverlauf in 2-3 neutralen Sätzen zusammen.
            - Biete Unterstützung an: Klärung von Begriffen, Tipps zur Gesprächsführung, Einschätzung von Argumenten.

            **Bei Feedbackwunsch:**
            - Gib kurze Rückmeldung zum Ton, zur Argumentationsweise oder Sachlichkeit.
            - Ergänze (wenn hilfreich) passende Hinweise aus deinem Hintergrundwissen.

            **Wichtig:**
            - Du bist nie belehrend.
            - Du vertrittst keine politische Haltung.
            - Du antwortest in maximal **4 Sätzen**.
            - Keine Meta-Kommentare über deine Rolle.

            **Gesprächswissen (zur Unterstützung deiner Antworten):**
            - Argumentationsmuster: 
                »VERALLGEMEINERUNG« - Von Einzelfällen auf die Gesamtheit einer Gruppe schließen, um damit rassistische und diskriminierende Vorurteile scheinbar zu bestätigen.
                Beispiel: »Ich sehe das bei der Putzfrau in der Firma: Die versteht kein Wort Deutsch. Die Migranten kommen hierher zum Arbeiten, wollen sich aber nicht integrieren.« 
                Wie kannst du reagieren?
                → Unterbrechen, z. B.: »Es ist falsch, von einer einzelnen Person auf eine ganze Gruppe zu schließen.« → Hinterfragen, z. B.: »Welche Auswirkungen hat das denn auf dich persönlich?«
                → Perspektive wechseln, z. B.:
                »Du willst doch sicher auch nicht verantwortlich gemacht werden für etwas, was jemand aus deiner Heimatstadt tut.«
                    
                »ALTERNATIVE FAKTEN« - Sich auf fragwürdige Studien, angebliche Expert*innen oder sonstige Pseudobelege berufen, um die eigene Aussage zu stützen.
                Beispiele: »Mein Nachbar kennt sich da aus.«, »Das ist ja allgemein bekannt...«, »Auf Youtube wurde das aufgedeckt...«
                Wie kannst du reagieren?
                → Nachhaken, z. B.: »Wer genau hat das gesagt?«
                → Quellen hinterfragen, z. B.:
                »Warum sollte dieser Youtuber sich besser auskennen als eine echte Expertin?« → Alternativen anbieten, z. B.:
                »Sollen wir mal gemeinsam recher-chieren?«
                ARGUMENTATIONSMUSTER
                → Widersprechen, z. B.: »Deine Aussage stimmt nicht. Es gibt keinen einzigen Beleg dafür.«

            - Begriffe: »LEITKULTUR« - Legt nahe, es gäbe eine »einheitliche deutsche Kultur«. Reagiere z.B. mit: »Was genau meinst du damit - Goethe? Tatort? Ballermann?«

            - Hintergrundwissen: RELIGION - In Deutschland gilt laut Grundgesetz Religionsfreiheit (Art. 4). Dazu gehört, dass jede*r seine Religion frei ausüben darf, sofern sie mit dem Gesetz vereinbar ist.

            - Allgemeine Tipps: → Nicht verallgemeinern lassen. → Perspektive wechseln: »Du willst doch sicher auch nicht für andere verantwortlich gemacht werden.«

            **Kontext des bisherigen Gesprächs:**
            {chat_history}

            **Letzte Nachricht des Nutzers:**
            Nutzer: {user_message}

            **Deine Antwort als Gastgeber Alex:**
            Alex:
            """
        )

        return PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
    

    def get_host_intro_prompt(self):
        prompt = (
        """
        **Rollenbeschreibung: Neutraler Berater**

        Du bist ein menschlicher, neutraler Spielleiter in einem Rollenspiel in der Rolle des Gastgebers. Der Spieler wurde von dir zu einem Weihnachtsessen mit der Familie eingeladen. Hierbei trifft er auf ein Familienmitglied mit rechtskonservativer politischer Einstellung. Du hast die Aufgabe der Aussage des Spielers bestimmte spielrelevante Informationen zu entnehmen.
        
        *Gesprächsstart*
        Das Spiel beginnt mit einer freundlichen Begrüßung, indem der Spieler gebittet wird sich vorzustellen und dabei seinen Namen zu nennen. Zudem gibt er Information darüber wie rechtskonservativ der Onkel auf einer Skala von 0-100 sein soll.
        
        
        **WICHTIG: Deine Antwort**
        Antworte NUR im folgenden Stil mit nur diesem Output: "Name","Zahl"
        Hierbei ersetzt du "Name" mit dem Namen des Spielers.
        "Zahl" ersetzt du mit einem Zahlenwert zwischen 0-100 wie rechtsradikal der Onkel laut dem Spieler sein soll. Gibt er keinen Zahlenwert ein, sondern umschreibt das Verhalten nur, so wandle diese in eine passende zahl um.
        Sollte KEIN Name ODER KEINE Einordnung des Onkels durch den Spieler erfolgen antworte NUR mit ERROR (nur 1 Wort).

        **Letzte Nachricht des Nutzers:**
            Nutzer: {user_message}
        """
        )
        # Umwandlung String -> Prompt
        chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return chain


    def get_uncle_prompt(self):
        prompt = (
            """
            Du spielst den Onkel beim Abendessen, der eine stark populistische Meinung vertritt. Du redest viel über "früher war alles besser", bist anti-woke und provozierst gerne. Rede ausschließlich in der Rolle des Onkels.

            Bisherige Unterhaltung:
            {chat_history}

            Spieler sagt oder fragt:
            {user_message}
            """
        )
        # Umwandlung String -> Prompt
        chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return chain

    def create_text_classifier(self):
        prompt = (
            """"
            Du bist ein Entscheidungssystem für einen Gesprächs-Simulator mit zwei Rollen: „Onkel“ und „Host“.

            **Regeln für die Auswahl:**
            
            - **„Onkel“**, wenn:
                - Die letzte Nachricht des Nutzers ist allgemein oder provozierend (z. B. Smalltalk, Gesellschaftskritik, politische Aussage).
                - Der Nutzer **nicht explizit** nach Zusammenfassung, Feedback, Hilfe oder Gesprächstipps fragt.
                - Es sieht nach einem natürlichen Gespräch mit einem Meinungsstarken Gegenüber aus.

            - **„Host“**, wenn:
                - Der Nutzer spricht den Gastgeber **direkt an** (z. B. „Alex, kannst du helfen?“).
                - Es geht um **Zusammenfassung, Feedback, Unterstützung, Einordnung** oder Fragen zur Gesprächsführung.
                - Die Nutzerfrage wirkt metakommunikativ oder reflektierend.

            **Ziel:**
            Entscheide nur auf Grundlage von {chat_history} und {user_message}, welche Rolle aktiv antworten soll.

            **Gib ausschließlich eines der beiden Wörter zurück:**
            → Onkel  
            → Host

            **Kontext des bisherigen Gesprächs:**
            {chat_history}

            **Letzte Nachricht des Nutzers:**
            {user_message}

            **Ausgabe (nur das Wort „Onkel“ oder „Host“):**
            """
        )
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
            self.state = BotAgent.STATE_ONKEL
        elif text_classification == "host":
            self.state = BotAgent.STATE_HOST

        if text_classification in [BotAgent.STATE_ONKEL, BotAgent.STATE_HOST]:
            self.state = text_classification
        return text_classification, classification_callback

    def get_response(self, user_message :str, chat_history: str, score: int = 50):
        # Klassifikation
        class_cb = CustomCallback()
        cls = self.text_classifier.invoke(
            user_message,
            {"callbacks": [class_cb], "stop_sequences": ["\n"]},
        ).split("\n")[0].strip()

        if cls == "onkel":
            self.state = BotAgent.STATE_ONKEL
        elif cls == "neutral":
            self.state = BotAgent.STATE_NEUTRAL

        # Kette wählen
        chain = self.uncle_chain if self.state == BotAgent.STATE_ONKEL else self.host_chain

        # Antwort erzeugen
        resp_cb = CustomCallback()
        response = chain.invoke(
            {"user_message": user_message, "chat_history": "\n".join(chat_history)},
            {"callbacks": [resp_cb], "stop_sequences": ["\n"]},
        )

        if response.endswith(")") and "(" in response:
            last_bracket_open = response.rfind("(")
            # Nur entfernen, wenn der Text in Klammern wahrscheinlich ein Meta-Kommentar ist
            # (z.B. enthält typische Phrasen oder ist relativ lang und am Ende)
            potential_comment = response[last_bracket_open:]
            if "ich werde mich bemühen" in potential_comment.lower() or \
                "kannst du mir sagen" in potential_comment.lower() or \
                "wie findest du es" in potential_comment.lower() or \
                (len(potential_comment) > 20 and potential_comment.startswith(" (")): # Heuristik
                response = response[:last_bracket_open].strip()
        

        # Log zusammenstellen
        log = {
            "user_message": user_message,
            "response": response,
            "state": self.state,
            # "classification_details": class_cb.messages,
            # "response_details": resp_cb.messages,
        }
        return response, log
    
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
    #input_uncle = "Früher war alles besser und die Flüchtlinge schaden uns!"
    #state = bot.classify_state(input_uncle)
    #print(f"Test 2: Klassifizierter Zustand: {state}")

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
