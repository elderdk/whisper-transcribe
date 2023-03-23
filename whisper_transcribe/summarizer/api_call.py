import openai
from typing import Union
from helpers import count_tokens


class APICaller:
    """
        "api_key": "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        "model": "text-davinci-003",
        "temperature": 1.2,
        "max_allowed_tokens": 4096,
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: Union[float, int],
        max_tokens: int,
        prompt: str,
    ):
        openai.api_key = api_key

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.result = None

    def count_tokens(self) -> int:
        """Count the number of tokens in the prompt."""
        return count_tokens(self.prompt)

    def call_api(self):
        """Call the OpenAI API."""
        result = openai.Completion.create(
            model=self.model,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        self.result = result

    def get_text_result(self) -> str:
        """Get the text result from the API call."""
        if self.result is None:
            self.call_api()

        return self.result["choices"][0]["text"]

    def __str__(self) -> str:
        """Return the string representation of the object."""
        return str(self.result)
