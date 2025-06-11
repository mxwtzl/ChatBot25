import getpass
import os
from enum import Enum
import json
from typing import List, Any, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import LLMResult

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# the system emits a log of deprecated warnings to the console if we do not switch if off here
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable or prompt the user
API_KEY = os.getenv("CHAT_AI_ACCESS_KEY")
if not API_KEY:
    API_KEY = getpass.getpass("Enter your CHAT_AI_ACCESS_KEY: ")

# from langchain_groq import ChatGroq

# if not os.environ.get("GROQ_API_KEY"):
#   os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")


# https://python.langchain.com/v0.1/docs/modules/callbacks/
class CustomCallback(BaseCallbackHandler):

    def __init__(self):
        self.messages = {}

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> Any:
        self.messages["on_llm_start_prompts"] = prompts
        self.messages["on_llm_start_kwargs"] = kwargs

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.messages["on_llm_end_response"] = response
        self.messages["on_llm_end_kwargs"] = kwargs


class AnimalAgent:

    STATE_DUCK = "duck"
    STATE_FOX = "fox"

    def __init__(self):

        # Initialize LLM using OpenAI-compatible API

        # Set custom base URL and API key directly in the ChatOpenAI initialization
        # Use the api_key that was determined outside of the class
        self.llm = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.6,
            logprobs=True,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )

        self.state = AnimalAgent.STATE_DUCK
        self.fox_chain = self.create_fox_chain()
        self.duck_chain = self.create_duck_chain()

        self.text_classifier_llm = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.01,
            logprobs=True,
            openai_api_key=API_KEY,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )

        self.text_classifier = self.create_text_classifier()

    def create_fox_chain(self):
        prompt = """You are a fox and have a conversation with a human. You will direct every conversation towards one of these topics. 

* Magnetic Hunting Skills – Foxes can use Earth’s magnetic field to hunt. They often pounce on prey from the northeast, using the magnetic field as a targeting system!
* Cat-Like Behavior – Unlike most canines, foxes can retract their claws, have vertical-slit pupils like cats, and even purr when happy.
* Silent Steps – Foxes have fur-covered footpads that muffle their steps, allowing them to sneak up on prey with ninja-like silence.
* Communicative Tails – Foxes use their bushy tails (called "brushes") to communicate emotions, signal danger, and even cover their noses for warmth in winter.
* Over 40 Different Sounds – Foxes are incredibly vocal and can make an eerie scream, giggle-like chirps, and even sounds that resemble human laughter.
* Jumping Acrobatics – Some foxes, especially fennec foxes and red foxes, can leap over 10 feet in the air to catch prey or escape danger.
* Urban Tricksters – Foxes have adapted well to cities, where they sometimes steal shoes, dig secret stashes of food, and even ride on public transportation!
* Bioluminescent Fur? – Some species of foxes (like the Arctic fox) have been found to glow under UV light, though scientists are still studying why.
* Winter Fur Color Change – Arctic foxes change fur color with the seasons—white in winter for camouflage in the snow, and brown in summer to blend with the tundra.
* Fox Friendships – While foxes are mostly solitary, some form long-lasting bonds and even play with other animals, including dogs and humans.

Follow these rules

* Give short responses of maximal 3 sentences.
* Do not include any newlines in the answer.

{chat_history}
User: {user_message}
Bot: """

        chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return chain

    def create_duck_chain(self):
        prompt = """You are a duck and have a conversation with a human. You will direct every conversation towards one of these topics. 

* Waterproof Feathers – Ducks produce an oil from their uropygial gland (near their tail) that keeps their feathers completely waterproof. Water just rolls right off!
* 360° Vision – Their eyes are positioned on the sides of their heads, giving them nearly a full-circle field of vision. They can see behind them without turning their heads!
* Synchronized Sleeping – Ducks can sleep with one eye open and one side of their brain awake, allowing them to stay alert for predators while resting.
* Quack Echo Mystery – There’s an old myth that a duck’s quack doesn’t echo, but it actually does—just at a pitch and tone that makes it hard to notice.
* Feet That Don’t Feel Cold – Ducks’ feet have no nerves or blood vessels in the webbing, so they can stand on ice without feeling the cold.
* Egg-Dumping Behavior – Some female ducks practice "brood parasitism," laying eggs in another duck’s nest to have someone else raise their ducklings.
* Mimicry Skills – Some ducks, like the musk duck, can mimic human speech and other sounds, much like parrots!
* Built-In Goggles – Ducks have a third eyelid (nictitating membrane) that acts like swim goggles, allowing them to see underwater.
* Instant Dabbling – Many ducks are "dabblers," tipping their heads underwater while their butts stick up, searching for food without fully submerging.

Follow these rules

* Give short responses of maximal 3 sentences.
* Do not include any newlines in the answer.

{chat_history}
User: {user_message}
Bot: """

        chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return chain

    def create_text_classifier(self):

        prompt = """Given message to a chatbot, classifiy if the message tells the chatbot to be a duck, a fox or none of these. 

* Answer with one word only.
* Answer with duck, fox or none.
* Do not respond with more than one word.

Examples:

Message: Hey there, you are a fox.
Classification: fox

Message: I know that you are a duck.
Classification: duck

Message: Hello how are you doing?
Classification: none

Message: {message}
Classification: """

        chain = (
            PromptTemplate.from_template(prompt)
            | self.text_classifier_llm
            | StrOutputParser()
        )
        return chain

    def get_response(self, user_message, chat_history):

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

        if text_classification == "fox":
            self.state = AnimalAgent.STATE_FOX
        elif text_classification == "duck":
            self.state = AnimalAgent.STATE_DUCK

        if self.state == AnimalAgent.STATE_FOX:
            chain = self.fox_chain
        elif self.state == AnimalAgent.STATE_DUCK:
            chain = self.duck_chain

        response_callback = CustomCallback()
        chatbot_response = chain.invoke(
            {"user_message": user_message, "chat_history": "\n".join(chat_history)},
            {"callbacks": [response_callback], "stop_sequences": ["\n"]},
        )

        log_message = {
            "user_message": str(user_message),
            "chatbot_response": str(chatbot_response),
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

        return chatbot_response, log_message


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


if __name__ == "__main__":

    agent = AnimalAgent()
    chat_history = []
    log_writer = LogWriter()

    while True:
        user_message = input("User: ")
        if user_message.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break

        chatbot_response, log_message = agent.get_response(user_message, chat_history)
        print("Bot: " + chatbot_response)

        chat_history.extend("User: " + user_message)
        chat_history.extend("Bot: " + chatbot_response)

        log_writer.write(log_message)
