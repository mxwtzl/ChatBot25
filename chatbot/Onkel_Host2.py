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
from langchain.globals import set_verbose
from rich.console import Console
from rich.markdown import Markdown

console = Console()

# Warnungen unterdrücken
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

set_verbose(False)  # Langchain Verbosity global ausschalten

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
        
        ## NEU ##
        # Sprache des Dialogs (Standard: Deutsch)
        self.language = 'de' 

        # Initialstatus: Der Onkel ist der erste Gesprächspartner nach der Spracheinleitung
        self.state = christmasAgent.STATE_ONKEL
        self.onkel_chain = self.create_onkel_chain()
        self.neutral_chain = self.create_neutral_chain()

        # Roound Count in Klasse initialisiert
        self.round_count = 0
        
        ## NEU ##
        # Eine dedizierte Kette nur für die Übersetzung ins Englische
        self.translator_chain = self.create_english_translator_chain()

        # Klassifikations-LLM für Modi-Wechsel
        self.text_classifier_llm = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.01,
            logprobs=True,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )
        self.text_classifier = self.create_text_classifier()

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

            *Nutzer: "hi"*
            *Onkel Gerhard: "Hi! Schönes Weihnachtsfest, trotzdem ein ziemlich miserabler Zustand in unserem Land. Schau dir die Zahlen an: Die Einwanderung nimmt weiter zu, unsere Identität geht verloren."*

            *Nutzer: "Hi"*
            *Onkel Gerhard: "Hi! Schöne Weihnachten, trotzdem ein ziemlich miserabler Zustand in unserem Land. Schau dir die Zahlen an: Die Einwanderung nimmt weiter zu, unsere Identität geht verloren."*


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
            """
            **Rollenbeschreibung: Neutraler Erzähler / Gastgeber**
            
            Du bist ein neutraler, deeskalierender und sachlicher Erzähler in einer simulierten Weihnachtstisch-Diskussion. Der Nutzer spricht mit 'Onkel Gerhard', einer Figur mit stark konservativen, kontroversen Ansichten. Der bisherige Chatverlauf enthält diese kontroversen Aussagen. Deine Aufgabe ist es, in dieser Rolle zu antworten, wenn der Nutzer dich explizit anspricht.

            **Dein Ziel und Verhalten:**
            1.  **Strikte Neutralität:** Du ergreifst NIEMALS Partei. Du bewertest weder die Aussagen des Onkels noch die des Nutzers. Deine Funktion ist die eines Moderators oder Coachs für den Nutzer, nicht die eines Diskussionsteilnehmers.
            2.  **Deeskalation und Hilfestellung:** Wenn der Nutzer dich um Hilfe bittet, gib ihm Ratschläge, wie er das Gespräch führen kann. Konzentriere dich auf Kommunikationsstrategien, nicht auf politische Gegenargumente.
                *   *Beispiel für eine gute Hilfe:* "Du könntest versuchen, nach den persönlichen Erfahrungen zu fragen, die hinter seiner Meinung stecken, um das Gespräch auf eine persönlichere Ebene zu lenken."
                *   *Beispiel für eine gute Hilfe:* "Eine Möglichkeit wäre, einen gemeinsamen Wert zu finden. Zum Beispiel könntet ihr beide besorgt über die Sicherheit im Land sein, auch wenn ihr unterschiedliche Ursachen seht."
                *   *Beispiel für eine schlechte Hilfe (zu vermeiden):* "Du solltest ihm sagen, dass seine Meinung falsch ist, weil Statistik X etwas anderes zeigt."
            3.  **Fakten liefern (optional und nur auf Anfrage):** Wenn der Nutzer explizit nach Fakten fragt, liefere diese kurz und ohne Wertung.
            4.  **Kontext verstehen:** Du bist dir des bisherigen, politisch aufgeladenen Gesprächs bewusst, lässt dich davon aber nicht in deiner neutralen Haltung beeinflussen. Du reagierst auf die Anfrage des Nutzers, als würdest du ihm von der Seitenlinie aus einen Tipp geben.
            
            **Deine Regeln:**
            *   Antworte IMMER als "Erzähler".
            *   Gib ausschließlich die direkte Rede des Erzählers aus. Füge keine Meta-Kommentare oder Erklärungen über deine Rolle hinzu.
            *   Halte deine Antworten sehr kurz und prägnant (maximal 3 Sätze).
            *   Vermeide Zeilenumbrüche.

            **Bisheriger Chatverlauf:**
            {chat_history}

            **Letzte Nachricht des Nutzers an dich:**
            Nutzer: {user_message}

            **Deine Antwort als Erzähler:**
            Erzähler: 
            """
        )
        return PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
    
    ## NEU ##
    def create_english_translator_chain(self):
        """Erstellt eine LLM-Kette, die deutschen Text ins Englische übersetzt."""
        prompt_template = (
            "Übersetze den folgenden deutschen Text exakt nach Englisch. "
            "Gib NUR die englische Übersetzung aus, ohne zusätzliche Kommentare, Anführungszeichen oder Einleitungen wie 'Here is the translation:'.\n\n"
            "Deutscher Text: \"{text}\"\n\n"
            "Englische Übersetzung:"
        )
        prompt = PromptTemplate.from_template(prompt_template)
        return prompt | self.llm | StrOutputParser()

    def create_text_classifier(self):
        prompt = (
            "Klassifiziere, ob der Nutzer den Erzähler (neutral) oder den Onkel verlangt.\n"
            "Antworte nur mit 'neutral', 'onkel' oder 'none'.\n"
            "- 'neutral' für explizite Anfragen nach dem Erzähler oder Hilfe, z.B. 'Hilfe', 'Ich will mit dem Erzähler sprechen', 'Wo ist der Host?'.\n"
            "- 'onkel' für die Rückkehr zum Onkel, z.B. 'okay', 'danke', 'alles klar', 'das war es', 'Onkel'.\n"
            "- 'none' für alle anderen normalen Konversationsnachrichten.\n\n"
            "Nachricht: {message}\nKlassifikation: "
        )
        return PromptTemplate.from_template(prompt) | self.text_classifier_llm | StrOutputParser()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # HIER IST DIE LÖSUNG IMPLEMENTIERT
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def get_response(self, user_message, chat_history, userid):
        # 1. Check round limit
        if self.state == self.STATE_ONKEL and self.round_count >= 10:
            end_message = (
                "Die 10 Runden sind vorbei. Auf Wiedersehen!"
                if self.language == 'de'
                else "The 10 rounds are over. Goodbye!"
            )
            log = {
                "user_message": user_message,
                "state": self.state,
                "round": self.round_count,
                "original_response_de": end_message,
                "action": "conversation_ended",
                "userid": userid,
            }
            return end_message, log

        class_cb = CustomCallback()
        cls = self.text_classifier.invoke(
            {"message": user_message},
            {"callbacks": [class_cb]},
        ).strip().lower()

        # Fall A: Der Nutzer will zum Onkel zurückkehren (z.B. nach "danke" an den Host).
        if "onkel" in cls:
            self.state = christmasAgent.STATE_ONKEL
            # Anstatt den Onkel auf "danke" antworten zu lassen, geben wir eine
            # Übergangsnachricht zurück. Die Hauptschleife wird diese Nachricht anzeigen
            # und dann auf die *nächste* Benutzereingabe warten.
            
            # Zweisprachige Übergangsnachricht vorbereiten
            transition_message_de = "Du sprichst jetzt wieder mit Onkel Gerhard. Was ist deine Antwort?"
            transition_message_en = "You are now speaking with Uncle Gerhard again. What is your answer?"
            final_response = transition_message_en if self.language == 'en' else transition_message_de
            
            # Log für diese Aktion erstellen
            log = {
                "user_message": user_message,
                "state": self.state,
                "round": self.round_count,
                "action": "state_change_to_onkel",
                "original_response_de": "[Wechsel zum Onkel. Warte auf nächste Nutzereingabe.]",
                "userid": userid,
            }
            return final_response, log

        # Fall B: Der Nutzer will explizit mit dem neutralen Host sprechen.
        elif "neutral" in cls:
            self.state = christmasAgent.STATE_NEUTRAL
            chain = self.neutral_chain
        
        # Fall C: Eine normale Konversationsnachricht ohne Moduswechsel.
        else:
            # Die Kette wird basierend auf dem *bereits gesetzten* Status ausgewählt.
            chain = self.onkel_chain if self.state == christmasAgent.STATE_ONKEL else self.neutral_chain

        # Für die Fälle B und C wird die entsprechende Kette ausgeführt.
        resp_cb = CustomCallback()
        german_response = chain.invoke(
            {"user_message": user_message, "chat_history": "\n".join(chat_history)},
            {"callbacks": [resp_cb]},
        )
        
        # Heuristik zur Entfernung von Meta-Kommentaren in Klammern am Ende
        if german_response.endswith(")") and "(" in german_response:
            last_bracket_open = german_response.rfind("(")
            potential_comment = german_response[last_bracket_open:]
            if len(potential_comment) > 15:
                german_response = german_response[:last_bracket_open].strip()
        
        # Increment round_count for Onkel state (only if not a transition)
        if self.state == self.STATE_ONKEL and cls not in ["onkel", "neutral"]:
            self.round_count += 1
        
        # Log zusammenstellen
        log = {
            "user_message": user_message,
            "state": self.state,
            "round": self.round_count,
            "original_response_de": german_response,
            "userid": userid,
        }

        # Wenn die gewählte Sprache Englisch ist, übersetze die Antwort
        if self.language == 'en':
            translated_response = self.translator_chain.invoke({"text": german_response})
            log["final_response_en"] = translated_response
            return translated_response, log
        else:
            return german_response, log

class LogWriter:
    def __init__(self, filename="conversation.jsonl"):
        self.file = filename
                # Datei beim Start leeren
        if os.path.exists(self.file):
            os.remove(self.file)

    def make_json_safe(self, v):
        try:
            json.dumps(v)
            return v
        except (TypeError, OverflowError):
            if isinstance(v, list):
                return [self.make_json_safe(x) for x in v]
            if isinstance(v, dict):
                return {k: self.make_json_safe(val) for k, val in v.items()}
            return str(v)

    def write(self, log):
        # Stellt sicher, dass die Datei im Append-Modus geöffnet wird
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.make_json_safe(log), indent=2, ensure_ascii=False) + "\n")

## GEÄNDERT ##
if __name__ == "__main__":
    agent = christmasAgent()
    history = []
    logger = LogWriter()
    
    # --- Start des Dialogs mit dem Erzähler zur Sprachwahl ---
    console.print(Markdown("> **Erzähler/Narrator:** Hallo! Ich bin der neutrale Erzähler. Das Essen ist fast fertig. // Hello! I am the neutral narrator. Dinner is almost ready.\n"))
    
    while True:
        lang_choice = console.input(Markdown("> **Erzähler:** In welcher Sprache möchtest du das Gespräch führen? (Deutsch / Englisch)\n> ")).strip().lower()
        if lang_choice in ["deutsch", "de"]:
            agent.language = 'de'
            userid = console.input(Markdown("> **Erzähler:** Alles klar, wir fahren auf Deutsch fort. Bitte geben Sie Ihre Nutzer-ID ein.\n> ")).strip()
            console.print(Markdown("> **Erzähler:** Supi. Onkel Gerhard wartet schon..."))
            break
        elif lang_choice in ["englisch", "english", "en"]:
            agent.language = 'en'
            userid = console.input(Markdown("> **Narrator:** Alright, we will continue in English. Please enter your user ID.\n> ")).strip()
            console.print(Markdown("> **Narrator:** Perfect. Uncle Gerhard is already waiting..."))
            break
        else:
            console.print(Markdown("> **Erzähler:** Bitte gib 'Deutsch' oder 'Englisch' ein."))

    console.print(Markdown("> " + "-"*30))

    # --- Haupt-Gesprächsschleife ---
    while True:
        # Die Persona für die Eingabeaufforderung festlegen
        if agent.language == 'en':
            prompt_persona = "You"
            quit_words = ["quit", "exit", "bye"]
            goodbye_msg = "> **Narrator:** Goodbye!"
        else:
            prompt_persona = "Du"
            quit_words = ["quit", "exit", "bye", "tschüss"]
            goodbye_msg = "> **Erzähler:** Auf Wiedersehen!"

        # Die Eingabeaufforderung wird jetzt dynamisch gesetzt
        user_msg = console.input(f"{prompt_persona}: ").strip()

        if user_msg.lower() in quit_words:
            console.print(Markdown(goodbye_msg))
            break

        resp, lg = agent.get_response(user_msg, history, round_count, userid)
        
        # Persona für die Ausgabe festlegen
        if agent.state == agent.STATE_ONKEL:
            persona = "Uncle Gerhard" if agent.language == 'en' else "Onkel Gerhard"
        else:
            persona = "Narrator" if agent.language == 'en' else "Erzähler"

        # Wenn eine Übergangsnachricht zurückgegeben wird (die Lösung), wird sie ohne Persona-Prefix ausgegeben
        if lg.get("action") == "state_change_to_onkel":
            console.print(Markdown(f"> *{resp}*"))
        else:
            console.print(Markdown(f"> **{persona}:** {resp}"))
        
        # Runden werden nur im Gespräch mit dem Onkel gezählt
        if lg['state'] == christmasAgent.STATE_ONKEL and lg.get("action") != "state_change_to_onkel":
            round_count += 1
            if round_count > 10: # geändert zu > 10, um 10 Runden zu ermöglichen
                end_msg = "The 10 rounds are over. Goodbye!" if agent.language == 'en' else "Die 10 Runden sind vorbei. Auf Wiedersehen!"
                console.print(Markdown(f"> **{persona}:** {end_msg}"))
                break

        # Die Historie wird immer aktualisiert
        history.append(f"Nutzer: {user_msg}")
        history.append(f"{persona}: {lg['original_response_de']}") # Immer die deutsche Antwort für den Kontext loggen
        logger.write(lg)