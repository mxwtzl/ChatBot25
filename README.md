# Demo for usage of Langchain with LLMs from chat-ai.academiccloud.de
## Overview

This is a simple demo that shows how the API from https://chat-ai.academiccloud.de/v1 can be used. 
It demonstrates how to interact with LLMs (Large Language Models) using Langchain.

For further information about Chat-AI, please also refer to the [Chat-AI.MD](/Chat-AI.md) document.

# Animalbot

The animalbot plays the role of a duck. He will only talk about duck related topics. You can tell him to be a fox, in which he will change his personality to being a fox. If you tell him to be a duck, he will change the personality back to duck.

## Example conversation

User: hi

Bot: Quack! Nice to meet you, human! Did you know that we ducks have waterproof feathers? We produce an oil from our uropygial gland that keeps our feathers completely waterproof, allowing us to swim and dive without getting wet!

User: tell me about your feathers

Bot: My feathers! ...

User: you are a fox

Bot: My apologies! As a fox, I'd like to tell you about my silent steps. My fur-covered footpads are specially designed to muffle my steps, allowing me to sneak up on prey with ninja-like silence. It's a vital skill for a hunter like me!

User: what kind of animal yare you?

Bot: I'm a fox! And did you know that, unlike most canines, I can retract my claws, have vertical-slit pupils like cats, and even purr when I'm happy?

User: what? tell me about your claws

Bot: My claws! Unlike most dogs, I have retractable claws that...

## How to run it

* Install the requirements.txt
* `python animalbot.py` to start the chat on the console

## Setting up a Virtual Environment

To run the Animalbot in an isolated environment, it's recommended to use a Python virtual environment (venv). Follow these steps:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setting up the Environment Variables

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open the `.env` file in the project directory.
3. Set the `CHAT_AI_ACCESS_KEY` variable to your API key:
   ```
   CHAT_AI_ACCESS_KEY=your_actual_api_key_here
   ```
4. Save the file. The application will use this key by default.

Alternatively, you can set the environment variable at the OS level:

- On Windows (Command Prompt):
  ```cmd
  set CHAT_AI_ACCESS_KEY=your_actual_api_key_here
  ```
- On macOS/Linux (Terminal):
  ```bash
  export CHAT_AI_ACCESS_KEY=your_actual_api_key_here
  ```
