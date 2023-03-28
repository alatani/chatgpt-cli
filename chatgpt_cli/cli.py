#!/bin/env python
from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path

import fire
import requests
import yaml
from halo import Halo
from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown

BASE_ENDPOINT = "https://api.openai.com/v1"

PRICING_RATE = {
    "gpt-3.5-turbo": {"prompt": 0.002, "completion": 0.002},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
}

# Initialize the token counters
# Initialize the console
console = Console()

class Message(BaseModel):
    role: str
    content: str

    @classmethod
    def from_dict(cls, message: dict) -> Message:
        return cls(**message)

    @classmethod
    def to_list(cls, messages: list[Message]) -> list[dict]:
        return [message.dict() for message in messages]

    @classmethod
    def from_list(cls, messages: list[dict]) -> list[Message]:
        return [cls(**message) for message in messages]


class ChatContext(BaseModel):
    config: dict
    logfilepath: Path

    message_separator:str = "\n\n"
    json_anchor:str = "\n%%===\n"

    def __init__(self, config:dict, filename: str|None = None) -> None:
        timestr = ""
        logdir = Path(config["chatlog"]["dir"]).expanduser()
        if filename is None:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{timestr}.md"
        super().__init__(config=config, logfilepath=logdir / filename)

    def resolve(self) -> list[Message]:
        assert isinstance(self.logfilepath, Path)
        messages = []

        # make empty file if not exists.
        if not self.logfilepath.exists():
            with open(self.logfilepath, "w") as f:
                pass

        with open(self.logfilepath) as f:
            content = f.read()

            anchor_idx = content.find(self.json_anchor)
            if anchor_idx >= 0:
                # json部分
                context_json = content[anchor_idx+len(self.json_anchor):]
                messages = Message.from_list(json.loads(context_json))
                for message in messages:
                    if message.role == "user":
                        console.print(">>> " + message.content.strip())
                        console.print()
                    if message.role == "assistant":
                        console.print(Markdown(message.content.strip()))
                        console.print()

        return messages

    def make_markdown(self, messages: list[Message]) -> str:
        body = ""
        body += f"{self.message_separator}".join([
            f"**{message.role}**: {message.content.strip()}"
            for message in messages
        ])
        return body

    def flush(self, messages: list[Message]) -> None:
        body = self.make_markdown(messages)
        body += f"{self.json_anchor}"
        body += json.dumps(Message.to_list(messages))

        with open(self.logfilepath, "w") as f:
            f.write(body)



def load_config(profile: str = "defualt") -> dict:
    """
    Read a YAML config file and returns it's content as a dictionary
    """
    config_file_candidates = [
        "~/.chatgpt.yaml",
        "config.yaml",
    ]
    config_file: Path | None = None
    for i in range(len(config_file_candidates)):
        config_file = Path(config_file_candidates[i]).expanduser()
        if config_file.exists():
            break

    if config_file is None:
        raise "No config file found."

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if not config.get("api-key", "").startswith("sk"):
        config["api-key"] = os.environ.get("OAI_SECRET_KEY", "fail")

    if not config.get("api-key", "").startswith("sk"):
        keyfile_path = Path(config["profiles"]["defualt"]["keyfile"]).expanduser()
        with open(keyfile_path) as keyfile:
            config["api-key"] = keyfile.read()

    profile = "defualt"
    while not config.get("api-key", "").startswith("sk"):
        config["api-key"] = input(
            "Enter your OpenAI Secret Key (should start with 'sk-')\n"
        )

    return config


class ChatGPTResponse(BaseModel):
    message: Message
    prompt_tokens: int
    completion_tokens: int

class ChatGPTClient(BaseModel):
    config: dict

    def get_response(self, messages: list[Message]) -> ChatGPTResponse:
        body = {"model": self.config["model"], "messages": Message.to_list(messages)}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config['api-key']}",
        }
        # return ChatGPTResponse(
        #     Message(role="assistant", content="OK"),
        #     20,10
        # )

        try:
            r = requests.post(
                f"{BASE_ENDPOINT}/chat/completions", headers=headers, json=body
            )
        except requests.ConnectionError:
            console.print("Connection error, try again...", style="red bold")
            messages.pop()
            raise KeyboardInterrupt
        except requests.Timeout:
            console.print("Connection timed out, try again...", style="red bold")
            messages.pop()
            raise KeyboardInterrupt

        if r.status_code == 200:
            response = r.json()

            message_response = response["choices"][0]["message"]
            usage_response = response["usage"]

            return ChatGPTResponse(
                message = Message(role="assistant", content=message_response['content'].strip()),
                prompt_tokens = usage_response["prompt_tokens"],
                completion_tokens = usage_response["completion_tokens"]
            )

        elif r.status_code == 400:
            response = r.json()
            if "error" in response:
                if response["error"]["code"] == "context_length_exceeded":
                    console.print("Maximum context length exceeded", style="red bold")
                    raise EOFError
                    # TODO: Develop a better strategy to manage this case
            console.print("Invalid request", style="bold red")
            raise EOFError

        elif r.status_code == 401:
            console.print("Invalid API Key", style="bold red")
            raise EOFError

        elif r.status_code == 429:
            console.print("Rate limit or maximum monthly limit exceeded", style="bold red")
            messages.pop()
            raise KeyboardInterrupt

        else:
            console.print(f"Unknown error, status code {r.status_code}", style="bold red")
            console.print(r.json())
            raise EOFError

class Expense(BaseModel):
    config: dict
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def display(self, model) -> None:
        """
        Given the model used, display total tokens used and estimated expense
        """
        total_expense = self.__calculate_expense(
            PRICING_RATE[model]["prompt"],
            PRICING_RATE[model]["completion"],
        )
        console.print(f"Total tokens used: [green bold]{self.total_tokens()}")
        console.print(f"Estimated expense: [green bold]${total_expense}")

    def __calculate_expense(self,
        prompt_pricing: float,
        completion_pricing: float,
    ) -> float:
        """
        Calculate the expense, given the number of tokens and the pricing rates
        """
        expense = ((self.prompt_tokens / 1000) * prompt_pricing) + (
            (self.completion_tokens / 1000) * completion_pricing
        )
        return round(expense, 6)


class ChatGptCli:
    def __init__(self, context: str|None = None):
        self.context = context

    def debug(self):
        console.log(Path("~/.chatgpt/config.yaml").expanduser().absolute())

        config = load_config()
        console.log(config)

    def run(self, profile: str = "default"):
        history = FileHistory(".history")
        session = PromptSession(history=history, multiline=True, auto_suggest=AutoSuggestFromHistory())

        # try:
        config = load_config(profile)

        #Run the display expense function when exiting the script
        expence = Expense(config=config)
        atexit.register(expence.display, model=config["model"])

        console.print("ChatGPT CLI", style="bold")
        console.print(f"Model in use: [green bold]{config['model']}")
        chat_context = ChatContext(config)
        messages = chat_context.resolve()

        # Context from the command line option
        if self.context:
            console.print(f"Context file: [green bold]{self.context}")
            with open(Path(self.context).expanduser()) as file:
                messages.append(Message(role="assistant", content=file.read().strip()))


        spinner = Halo(text='Waiting..', spinner='dots')
        chat_gpt_client = ChatGPTClient(config=config)
        while True:
            try:
                input_message = session.prompt(HTML(f"<b>[{expence.total_tokens()}] >>> </b>"))

                if input_message.strip().lower() == "/q":
                    raise EOFError
                if input_message.strip().lower() == "":
                    raise KeyboardInterrupt


                spinner.start()
                messages.append(Message(role="user", content=input_message))
                response: ChatGPTResponse = chat_gpt_client.get_response(messages)
                spinner.stop()

                console.print()
                console.print(Markdown(response.message.content))
                console.print()
                messages.append(response.message)
                expence.add(response.prompt_tokens, response.completion_tokens)

                chat_context.flush(messages)
            except KeyboardInterrupt:
                messages.pop()
                continue
            except EOFError:
                break

def main():
    fire.Fire(ChatGptCli)

if __name__ == "__main__":
    main()
