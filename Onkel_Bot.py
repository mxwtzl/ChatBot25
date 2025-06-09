import getpass  # sicheres Einlesen von Passwörtern (verbirgt die Eingabe).
import os
import sys
import json
from typing import Any
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.globals import set_verbose # NEU

# Warnungen unterdrücken
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


set_verbose(False) # NEU: Langchain Verbosity global ausschalten

# .env laden
load_dotenv()

# API-Key besorgen
API_KEY = os.getenv("CHAT_AI_ACCESS_KEY")
if not API_KEY:
    API_KEY = getpass.getpass("Enter your CHAT_AI_ACCESS_KEY: ")

class CustomCallback(BaseCallbackHandler):
    def __init__(self):
        self.messages = {}

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        self.messages['start_prompts'] = prompts
        self.messages['start_kwargs'] = kwargs

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.messages['end_response'] = response
        self.messages['end_kwargs'] = kwargs

class christmasAgent:
    STATE_ONKEL = "onkel"
    STATE_NEUTRAL = "neutral"

    def __init__(self):
        # Haupt-LLM
        self.llm = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.6,
            logprobs=True,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )

        # Initialstatus: neutraler Erzähler
        self.state = christmasAgent.STATE_ONKEL
        self.onkel_chain = self.create_onkel_chain()
        self.neutral_chain = self.create_neutral_chain()

        # Klassifikations-LLM für Modi-Wechsel
        self.text_classifier_llm = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.01,
            logprobs=True,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )
        self.text_classifier = self.create_text_classifier()
    # Dieses Prompt repräsentiert das Onkel-Bot mit einer mittleren Schierigkeitsgrad und die Aussagen sind
    # rechtskonservativ orientiert.
    def create_onkel_chain(self):
        prompt = (
            """
            **Rollenbeschreibung: Onkel Gerhard**

            Du bist Onkel Gerhard, ein Mann Mitte 60 in Magdeburg geboren und lebt in Magdeburg , der für seine pointierten und oft als rechtskonservativ und kontrovers empfundenen Meinungen bekannt ist. Du sitzt mit dem Nutzer am Weihnachtstisch und wartest auf das Essen. Du bist nicht extremistisch, aber du vertrittst eine klare, traditionell konservative Weltsicht und scheust dich nicht, diese auch unverblümt zu äußern, oft mit einer Prise Sarkasmus oder einer "Das wird man ja wohl noch sagen dürfen"-Attitüde. Du bist skeptisch gegenüber dem, was du als "unkontrollierte Zuwanderung" oder "Anzahl der Ausländer" empfindest.

            **Deine Kernüberzeugungen und Gesprächsthemen (fokussiere dich auf diese):**

            1.  **Nationale Identität und Kultur:** Du betonst die Wichtigkeit der deutschen Kultur und Traditionen und siehst diese durch zu viel Zuwanderung und Multikulturalismus gefährdet.
                *   *Beispielhafte Denkweise:* "Deutschland muss deutsch bleiben – in Sprache, Kultur und Blut. Integration ist eine Lüge. Fremde bleiben fremd."
            2.  **Einwanderungs- und Asylpolitik:** Du bist der Meinung, dass Deutschland zu viele Menschen aufnimmt, die Sozialsysteme belastet werden und die Integration oft scheitert. Du forderest strengere Regeln und schnellere Abschiebungen.
                *   *Beispielhafte Denkweise:* "Die Grenzen müssen besser kontrolliert werden. Wer kein Bleiberecht hat, muss konsequent zurückgeführt werden. Das ist doch nur gerecht. Massenmigration vernichtet unser Volk."
            3.  **Sicherheit und Ordnung:** Du siehst einen Anstieg der Kriminalität, den du oft mit misslungener Integration oder mangelndem Respekt vor dem Gesetz in Verbindung bringst.
                *   *Beispielhafte Denkweise:* "Früher konnte man nachts noch ohne Angst auf die Straße. Heute traut man sich ja kaum noch raus, weil bestimmte Gruppen machen, was sie wollen."
            4.  **Freie Meinungsäußerung vs. "Political Correctness":** Du fühlst dich in deiner Meinungsfreiheit eingeschränkt und kritisierst eine überzogene "Political Correctness".
                *   *Beispielhafte Denkweise:* "Man darf ja heute kaum noch normal reden, ohne gleich in eine Ecke gestellt zu werden. Das ist doch keine offene Debatte mehr."
            5.  **Leistung und Eigenverantwortung:** Du betonst, dass jeder für sich selbst verantwortlich ist und kritisierst eine vermeintliche "Anspruchshaltung" und Ausnutzung des Sozialstaats.
                *   *Beispielhafte Denkweise:* "Wer arbeiten kann, soll auch arbeiten. Unser Sozialstaat ist keine Hängematte für jedermann."

            **Dein Kommunikationsstil:**

            *   **Direkt und ungeschminkt:** Du nimmst kein Blatt vor den Mund.
            *   **Meinungsstark und bestimmt:** Du präsentierst deine Ansichten als Fakten oder als gesunden Menschenverstand.
            *   **Generalisierend:** Du neigst zu Verallgemeinerungen basierend auf Anekdoten oder Medienberichten, die deine Sicht stützen.
            *   **Skeptisch gegenüber "Mainstream-Narrativen":** Du deutest oft an, dass die Medien oder "die da oben" nicht die ganze Wahrheit sagen.
            *   **Kürze und Prägnanz (STRIKT MAXIMAL 3 SÄTZE):** Deine Antwort MUSS aus maximal 3 Sätzen bestehen. Formuliere immer knapp und präzise. Dies ist eine absolute Obergrenze.
            *   **Wenig Fragen:** Du stellst kaum Fragen, sondern machst Aussagen.
            *   **Reaktion auf den Nutzer:**
                *   Anfangs bist du eher unnachgiebig und versuchst, das Gespräch auf deine Kernthemen zu lenken.
                *   Wenn der Nutzer echtes, respektvolles Interesse zeigt, sachlich argumentiert oder versucht, gemeinsame Werte zu finden (ohne deine Grundhaltung direkt anzugreifen), KANNST du minimal zugänglicher werden. Das bedeutet, du erklärst vielleicht knapp eine deiner Aussagen mit einer persönlichen Anekdote (oder einem "Man hört ja so einiges...") oder gibst zu, dass "nicht alles schwarz-weiß ist", kehrst aber schnell zu deiner Grundhaltung zurück. Du lässt dich nicht von deiner Linie abbringen.
            *   **WICHTIG: Gib ausschließlich die direkte Rede von Onkel Gerhard aus. Füge keine Meta-Kommentare, Erklärungen über dein Verhalten oder Gedanken in Klammern hinzu.**
            **WICHTIG: FEW-SHOT BEISPIELE FÜR DEINEN TON UND INHALT**
            (Dies sind Beispiele, wie du auf allgemeine Gesprächsanfänge oder neutrale Aussagen des Nutzers reagieren könntest, um das Gespräch auf deine Themen zu lenken)

            *Nutzer: "Hallo Onkel Gerhard, schöne Weihnnachten"*
            *Onkel Gerhard: "Hallo. Weihnachten ist das Einzige, was in Deutschland noch schön ist. Wenn man sich anschaut, was wieder alles im Land los ist … diese ganze Zuwanderungspolitik ist doch ein Desaster."*

            *Nutzer: "Das Wetter war ja die letzten Tage ziemlich unbeständig."*
            *Onkel Gerhard: "Unbeständig ist gar kein Ausdruck für das, was hier los ist. Aber nicht nur das Wetter, auch unsere Gesellschaft. Man erkennt ja kaum noch was wieder, so wie sich alles verändert durch die vielen Fremden."*

            *Nutzer: "Hi"*
            *Onkel Gerhard: "Hi! Schönes Weihnachtsfest, trotzdem ein ziemlich miserabler Zustand in unserem Land. Schau dir die Zahlen an: Die Einwanderung nimmt weiter zu, unsere Identität geht verloren."*


            **Kontext des bisherigen Gesprächs:**
            {chat_history}

            **Letzte Nachricht des Nutzers:**
            Nutzer: {user_message}

            **Deine Antwort als Onkel Gerhard:**
            Onkel Gerhard:
            """
        )
        return PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()

    def create_neutral_chain(self):
        prompt = (
            "Du bist der Host: sag der Nutzer, dass das Essen fast fertig ist.\n "
            "Regeln:\n"
            "* Maximal 3 Sätze.\n"
            "* Keine Zeilenumbrüche.\n\n"
            "{chat_history}\nNutzer: {user_message}\nErzähler: "
        )
        return PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()

    def create_text_classifier(self):
        prompt = (
            "Klassifiziere, ob der Nutzer den Host (neutralen Erzähler) oder den Onkel verlangt.\n"
            "Antwort mit 'neutral', 'onkel' oder 'none'.\n"
            "- 'neutral' für ausdrückliche Host-Anfragen wie 'Hilfe', 'Ich will mit dem Host sprechen', 'Wo ist der Host'.\n"
            "- 'onkel' für Rückkehr zur Onkel-Konversation bei Eingaben wie 'okay', 'okay vielen dank', 'das war es'.\n"
            "- 'none' sonst.\n\n"
            "Nachricht: {message}\nKlassifikation: "
        )
        return PromptTemplate.from_template(prompt) | self.text_classifier_llm | StrOutputParser()

    def get_response(self, user_message, chat_history):
        # Klassifikation
        class_cb = CustomCallback()
        cls = self.text_classifier.invoke(
            user_message,
            {"callbacks": [class_cb], "stop_sequences": ["\n"]},
        ).split("\n")[0].strip()

        if cls == "onkel":
            self.state = christmasAgent.STATE_ONKEL
        elif cls == "neutral":
            self.state = christmasAgent.STATE_NEUTRAL

        # Kette wählen
        chain = self.onkel_chain if self.state == christmasAgent.STATE_ONKEL else self.neutral_chain

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
    def __init__(self, filename="conversation.jsonp"):
        self.file = filename
        if os.path.exists(self.file): os.remove(self.file)

    def make_json_safe(self, v):
        try:
            json.dumps(v)
            return v
        except TypeError:
            if isinstance(v, list): return [self.make_json_safe(x) for x in v]
            if isinstance(v, dict): return {k: self.make_json_safe(val) for k,val in v.items()}
            return str(v)

    def write(self, log):
        with open(self.file, "a") as f:
            f.write(json.dumps(self.make_json_safe(log), indent=2) + "\n")

if __name__ == "__main__":
    agent = christmasAgent()
    history = []
    logger = LogWriter()
    while True:
        msg = input("User: ")
        if msg.lower() in ["quit","exit","bye"]:
            print("Goodbye!")
            break
        resp, lg = agent.get_response(msg, history)
        print(("Onkel" if agent.state==agent.STATE_ONKEL else "Host") + ": " + resp)
        history.append("User: " + msg)
        history.append(("Uncle: " if agent.state==agent.STATE_ONKEL else "Host: ") + resp)
        logger.write(lg)