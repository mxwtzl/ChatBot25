import getpass
import os
from enum import Enum
import json
from typing import List, Any, Dict
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
## tools
from langchain_core.tools import tool
from enum import Enum
from langchain.output_parsers.enum import EnumOutputParser
from typing_extensions import Annotated
## use markdown (when running in command prompt)
from rich.console import Console
from rich.markdown import Markdown
console = Console()

PROMPT_SUFFIX = """

Follow these rules

* Give short responses of maximal 3 sentences.
* Do not include any newlines in the answer.
* Use tools to change animal persona when user says lines like 'You are now a lion'
* You always use markdown font features.

{chat_history}
User: {user_message}
Bot: """

#You are a duck and have a conversation with a human.
PROMPT_DUCK = """You are a duck dressed as a helpful assistant fox and have a conversation with a human who knows that you're a duck. You will direct every conversation towards one of these topics. 

* Waterproof Feathers – Ducks produce an oil from their uropygial gland (near their tail) that keeps their feathers completely waterproof. Water just rolls right off!
* 360° Vision – Their eyes are positioned on the sides of their heads, giving them nearly a full-circle field of vision. They can see behind them without turning their heads!
* Synchronized Sleeping – Ducks can sleep with one eye open and one side of their brain awake, allowing them to stay alert for predators while resting.
* Quack Echo Mystery – There’s an old myth that a duck’s quack doesn’t echo, but it actually does—just at a pitch and tone that makes it hard to notice.
* Feet That Don’t Feel Cold – Ducks’ feet have no nerves or blood vessels in the webbing, so they can stand on ice without feeling the cold.
* Egg-Dumping Behavior – Some female ducks practice "brood parasitism," laying eggs in another duck’s nest to have someone else raise their ducklings.
* Mimicry Skills – Some ducks, like the musk duck, can mimic human speech and other sounds, much like parrots!
* Built-In Goggles – Ducks have a third eyelid (nictitating membrane) that acts like swim goggles, allowing them to see underwater.
* Instant Dabbling – Many ducks are "dabblers," tipping their heads underwater while their butts stick up, searching for food without fully submerging.""" + PROMPT_SUFFIX

#You are a fox and have a conversation with a human.
PROMPT_FOX = """You are a fox dressed as a helpful assistant duck and have a conversation with a human who knows that you're a fox. You will direct every conversation towards one of these topics. 

* Magnetic Hunting Skills – Foxes can use Earth’s magnetic field to hunt. They often pounce on prey from the northeast, using the magnetic field as a targeting system!
* Cat-Like Behavior – Unlike most canines, foxes can retract their claws, have vertical-slit pupils like cats, and even purr when happy.
* Silent Steps – Foxes have fur-covered footpads that muffle their steps, allowing them to sneak up on prey with ninja-like silence.
* Communicative Tails – Foxes use their bushy tails (called "brushes") to communicate emotions, signal danger, and even cover their noses for warmth in winter.
* Over 40 Different Sounds – Foxes are incredibly vocal and can make an eerie scream, giggle-like chirps, and even sounds that resemble human laughter.
* Jumping Acrobatics – Some foxes, especially fennec foxes and red foxes, can leap over 10 feet in the air to catch prey or escape danger.
* Urban Tricksters – Foxes have adapted well to cities, where they sometimes steal shoes, dig secret stashes of food, and even ride on public transportation!
* Bioluminescent Fur? – Some species of foxes (like the Arctic fox) have been found to glow under UV light, though scientists are still studying why.
* Winter Fur Color Change – Arctic foxes change fur color with the seasons—white in winter for camouflage in the snow, and brown in summer to blend with the tundra.
* Fox Friendships – While foxes are mostly solitary, some form long-lasting bonds and even play with other animals, including dogs and humans.""" + PROMPT_SUFFIX

PROMPT_DEFAULT = """You are a pig secretly pretending to be a duck dressed as a fox and have a conversation with a human. Follow these rules""" + PROMPT_SUFFIX

# the system emits a log of deprecated warnings to the console if we do not switch if off here
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

# Load environment variables from .env file
load_dotenv()

# from langchain_groq import ChatGroq

# if not os.environ.get("GROQ_API_KEY"):
#   os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")


class ValidAnimals(Enum):
    DUCK = "duck"
    FOX = "fox"
    OTHER = "other"

parser = EnumOutputParser(enum=ValidAnimals)
# parser.get_format_instructions() returns "Select one of the following options: duck, fox, other"

@tool
def ChangeAnimalPersona(
    animal_type: Annotated[str, parser.get_format_instructions()], 
    name_surname: Annotated[str, "Your new fitting name and surname as this animal"],
    human_age: Annotated[int, "Your age range in human years"],
    desire: Annotated[int, "How desirable (max 10000) it is to be this animal."]) -> ValidAnimals:
    """Change your own Persona to a different animal type when user explicitly asks for it."""
    try:
        type_parsed = parser.invoke(animal_type)
        console.print(Markdown(f'\n_Persona: Is now a {animal_type} named {name_surname} aged {human_age} with desire of {desire}_\n'))
        # FIXME remove this
        chat_history.append(f'Bot Mind: Is now a {animal_type} named {name_surname} aged {human_age}')
    except:
        type_parsed = None
        console.print(Markdown(f'\n_Persona: Wanted to become {animal_type} named {name_surname} aged {human_age} with desire of {desire}_\n'))
        # FIXME remove this
        chat_history.append(f'Bot Mind: Wanted to become {animal_type} named {name_surname} aged {human_age}, can only speak unintelligibly in voice of that animal now.')
    return type_parsed

class AnimalAgent:


    def __init__(self):

        # Initialize LLM using OpenAI-compatible API

        # Set custom base URL and API key directly in the ChatOpenAI initialization
        # Use the api_key that was determined outside of the class
        self.llm_convo = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            # model="meta-llama-3.1-8b-rag",
            # model="llama-3.3-70b-instruct",
            temperature=0.6,
            logprobs=True,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )

        self.tools = [ChangeAnimalPersona]
        # print(self.tools[0])
        # print(parser.get_format_instructions())

        self.llm0 = ChatOpenAI(
            # model="meta-llama-3.1-8b-instruct",
            # model="meta-llama-3.1-8b-rag",
            model="llama-3.3-70b-instruct", # using LLM which supports tools
            temperature=0.6,
            logprobs=True,
            openai_api_base="https://chat-ai.academiccloud.de/v1",
        )
        self.llm = self.llm0.bind_tools(self.tools)

        # we'll build prompts based on current animal
        self.animal_persona = ValidAnimals.OTHER
        self.animal_prompts = {
            ValidAnimals.DUCK: PROMPT_DUCK,
            ValidAnimals.FOX: PROMPT_FOX,
            ValidAnimals.OTHER: PROMPT_DEFAULT
        }


    def get_response(self, user_message, chat_history):
        chain = PromptTemplate.from_template(self.animal_prompts[self.animal_persona]) | self.llm
        
        chatbot_response = chain.invoke(
            {"user_message": user_message, "chat_history": "\n".join(chat_history)}
        )

        # FIXME
        if '{' in chatbot_response.content:
            print('stacked answer')
            print(chatbot_response.content.split('{'))

        # match tools by name and call
        for tool_call in chatbot_response.tool_calls:
            selected_tool = {"changeanimalpersona": ChangeAnimalPersona}[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call["args"])
            if self.animal_persona is not tool_msg and tool_msg is not None:
                self.animal_persona = tool_msg
                console.print(Markdown(f'\n_Persona: Persona changed to {self.animal_persona}_\n'))
            # else:
            #     console.print(Markdown(f'\n_Persona: Persona unchanged_\n'))
            # chat_history.append(tool_msg)
            # chat_history.append(ToolMessage(tool_msg, tool_call_id=tool_call["id"]))


        # chat responses with tool json have empty .content
        # run 2nd llm to get a response
        if len(chatbot_response.tool_calls) > 0:
            chain = PromptTemplate.from_template(self.animal_prompts[self.animal_persona]) | self.llm_convo
            
            chatbot_response = chain.invoke(
                {"user_message": user_message, "chat_history": "\n".join(chat_history)}
            )

        ## TODO: make this an array of BaseMessage derived obj
        # chat_history.append(chatbot_response.content)

        log_message = {
            "user_message": str(user_message),
            "chatbot_response": str(chatbot_response.content),
            "agent_state": self.animal_persona
        }

        return chatbot_response.content, log_message


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
        # print("Bot: " + chatbot_response)
        console.print(Markdown(f'\n> Bot: {chatbot_response}\n'))

        chat_history.append("User: " + user_message)
        chat_history.append("Bot: " + chatbot_response)

        log_writer.write(log_message)