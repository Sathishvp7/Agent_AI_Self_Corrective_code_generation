import os
from pprint import pprint
from IPython.display import display_markdown
from openai import OpenAI

# @title
"""
This is a collection of helper functions and methods we are going to use in
the Agent implementation. You don't need to know the specific implementation
of these to follow the Agent code. But, if you are curious, feel free to check
them out.
"""

import time

from colorama import Fore
from colorama import Style


def completions_create(client, messages: list, model: str) -> str:
    """
    Sends a request to the client's `completions.create` method to interact with the language model.

    Args:
        client (OpenAI): The OpenAI client object
        messages (list[dict]): A list of message objects containing chat history for the model.
        model (str): The model to use for generating tool calls and responses.

    Returns:
        str: The content of the model's response.
    """
    response = client.chat.completions.create(messages=messages, model=model)
    return str(response.choices[0].message.content)


def build_prompt_structure(prompt: str, role: str, tag: str = "") -> dict:
    """
    Builds a structured prompt that includes the role and content.

    Args:
        prompt (str): The actual content of the prompt.
        role (str): The role of the speaker (e.g., user, assistant).

    Returns:
        dict: A dictionary representing the structured prompt.
    """
    if tag:
        prompt = f"<{tag}>{prompt}</{tag}>"
    return {"role": role, "content": prompt}

def update_chat_history(history: list, msg: str, role: str):
    """
    Updates the chat history by appending the latest response.

    Args:
        history (list): The list representing the current chat history.
        msg (str): The message to append.
        role (str): The role type (e.g. 'user', 'assistant', 'system')
    """
    history.append(build_prompt_structure(prompt=msg, role=role))


class ChatHistory(list):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        if messages is None:
            messages = []

        super().__init__(messages)
        self.total_length = total_length

    def append(self, msg: str):
        """Add a message to the queue.

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(0)
        super().append(msg)



class FixedFirstChatHistory(ChatHistory):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        super().__init__(messages, total_length)

    def append(self, msg: str):
        """Add a message to the queue. The first messaage will always stay fixed.

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(1)
        super().append(msg)

def fancy_print(message: str) -> None:
    """
    Displays a fancy print message.

    Args:
        message (str): The message to display.
    """
    print(Style.BRIGHT + Fore.CYAN + f"\n{'=' * 50}")
    print(Fore.MAGENTA + f"{message}")
    print(Style.BRIGHT + Fore.CYAN + f"{'=' * 50}\n")
    time.sleep(0.5)


def fancy_step_tracker(step: int, total_steps: int) -> None:
    """
    Displays a fancy step tracker for each iteration of the generation-reflection loop.

    Args:
        step (int): The current step in the loop.
        total_steps (int): The total number of steps in the loop.
    """
    fancy_print(f"STEP {step + 1}/{total_steps}")

BASE_GENERATION_SYSTEM_PROMPT = """
Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempt.
You must always output the revised content.
"""

BASE_REFLECTION_SYSTEM_PROMPT = """
You are tasked with generating critique and recommendations to the user's generated content.
If the user content has something wrong or something to be improved, output a list of recommendations
and critiques.
"""


class ReflectionAgent:
    """
    A class that implements a Reflection Agent, which generates responses and reflects
    on them using the LLM to iteratively improve the interaction. The agent first generates
    responses based on provided prompts and then critiques them in a reflection step.

    Attributes:
        model (str): The model name used for generating and reflecting on responses.
        client (OpenAI): An instance of the OpenAI client to interact with the language model.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model

    def _request_completion(
        self,
        history: list,
        verbose: int = 0,
        log_title: str = "COMPLETION",
        log_color: str = "",
    ):
        """
        A private method to request a completion from the OpenAI model.

        Args:
            history (list): A list of messages forming the conversation or reflection history.
            verbose (int, optional): The verbosity level. Defaults to 0 (no output).

        Returns:
            str: The model-generated response.
        """
        output = completions_create(self.client, history, self.model)

        if verbose > 0:
            print(log_color, f"\n\n{log_title}\n\n", output)

        return output

    def generate(self, generation_history: list, verbose: int = 0) -> str:
        """
        Generates a response based on the provided generation history using the model.

        Args:
            generation_history (list): A list of messages forming the conversation or generation history.
            verbose (int, optional): The verbosity level, controlling printed output. Defaults to 0.

        Returns:
            str: The generated response.
        """
        return self._request_completion(
            generation_history, verbose, log_title="GENERATION", log_color=Fore.BLUE
        )

    def reflect(self, reflection_history: list, verbose: int = 0) -> str:
        """
        Reflects on the generation history by generating a critique or feedback.

        Args:
            reflection_history (list): A list of messages forming the reflection history, typically based on
                                       the previous generation or interaction.
            verbose (int, optional): The verbosity level, controlling printed output. Defaults to 0.

        Returns:
            str: The critique or reflection response from the model.
        """
        return self._request_completion(
            reflection_history, verbose, log_title="REFLECTION", log_color=Fore.GREEN
        )

    def run(
        self,
        user_msg: str,
        generation_system_prompt: str = "",
        reflection_system_prompt: str = "",
        n_steps: int = 10,
        verbose: int = 0,
    ) -> str:
        """
        Runs the ReflectionAgent over multiple steps, alternating between generating a response
        and reflecting on it for the specified number of steps.

        Args:
            user_msg (str): The user message or query that initiates the interaction.
            generation_system_prompt (str, optional): The system prompt for guiding the generation process.
            reflection_system_prompt (str, optional): The system prompt for guiding the reflection process.
            n_steps (int, optional): The number of generate-reflect cycles to perform. Defaults to 3.
            verbose (int, optional): The verbosity level controlling printed output. Defaults to 0.

        Returns:
            str: The final generated response after all cycles are completed.
        """
        generation_system_prompt += BASE_GENERATION_SYSTEM_PROMPT
        reflection_system_prompt += BASE_REFLECTION_SYSTEM_PROMPT

        # Given the iterative nature of the Reflection Pattern, we might exhaust the LLM context (or
        # make it really slow). That's the reason I'm limitting the chat history to three messages.
        # The `FixedFirstChatHistory` is a very simple class, that creates a Queue that always keeps
        # fixed the first message. I thought this would be useful for maintaining the system prompt
        # in the chat history.
        generation_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=generation_system_prompt, role="system"),
                build_prompt_structure(prompt=user_msg, role="user"),
            ],
            total_length=3,
        )

        reflection_history = FixedFirstChatHistory(
            [build_prompt_structure(prompt=reflection_system_prompt, role="system")],
            total_length=3,
        )

        for step in range(n_steps):
            if verbose > 0:
                fancy_step_tracker(step, n_steps)

            # Generate the response
            generation = self.generate(generation_history, verbose=verbose)
            update_chat_history(generation_history, generation, "assistant")
            update_chat_history(reflection_history, generation, "user")

            # Reflect and critique the generation
            critique = self.reflect(reflection_history, verbose=verbose)

            update_chat_history(generation_history, critique, "user")
            update_chat_history(reflection_history, critique, "assistant")

        return generation

agent = ReflectionAgent()

generation_system_prompt = "You are a Python programmer tasked with generating high quality Python code"

reflection_system_prompt = "You are Python genius, an experienced computer scientist"

user_msg = "Generate a Python implementation of the Merge Sort algorithm"

final_response = agent.run(
    user_msg=user_msg,
    generation_system_prompt=generation_system_prompt,
    reflection_system_prompt=reflection_system_prompt,
    n_steps=3,
    verbose=1,
)
print(final_response)