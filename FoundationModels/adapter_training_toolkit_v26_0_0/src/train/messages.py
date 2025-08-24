from enum import Enum
from typing import Literal, TypeAlias, TypedDict


class Locale(str, Enum):
    """Supported locale."""

    EN_US = "en_US"
    EN_GB = "en_GB"
    EN_AU = "en_AU"
    EN_CA = "en_CA"
    EN_IE = "en_IE"
    EN_NZ = "en_NZ"
    EN_ZA = "en_ZA"
    EN_IN = "en_IN"
    EN_SG = "en_SG"
    PT_BR = "pt_BR"
    PT_PT = "pt_PT"
    FR_FR = "fr_FR"
    FR_CA = "fr_CA"
    FR_BE = "fr_BE"
    FR_CH = "fr_CH"
    DE_DE = "de_DE"
    DE_CH = "de_CH"
    DE_AT = "de_AT"
    IT_IT = "it_IT"
    IT_CH = "IT_CH"
    ES_ES = "es_ES"
    ES_US = "es_US"
    ES_MX = "es_MX"
    ES_CL = "es_CL"
    ZH_CN = "zh_CN"
    ZH_TW = "zh_TW"
    ZH_HK = "zh_HK"
    JA_JP = "ja_JP"
    KO_KR = "ko_KR"


class Message(TypedDict):
    """
    A class representing an individual message, for example:
    ```
    {"role": "user", "content": "<USER PROMPT>"}
    ```
    """

    role: Literal["system", "user", "assistant"]
    """The authors role, one-of "system", "user", "assistant"."""

    content: str
    """The content of the message."""

    @staticmethod
    def default_system_message(locale: Locale = Locale.EN_US) -> "Message":
        """Returns the default system message for the given locale."""

        if locale == Locale.EN_US:
            content = "A conversation between a user and a helpful assistant."
        elif locale == Locale.EN_GB:
            content = "A conversation between a user located in Britain and a helpful assistant. The assistant is mindful of British spelling, vocabulary, entities and other British context."  # noqa: E501
        elif locale == Locale.EN_AU:
            content = "A conversation between a user located in Australia and a helpful assistant. The assistant is mindful of Australian spelling, vocabulary, entities and other Australian context."  # noqa: E501
        elif locale == Locale.EN_CA:
            content = "A conversation between a user located in Canada and a helpful assistant. The assistant is mindful of Canadian spelling, vocabulary, entities and other Canadian context."  # noqa: E501
        elif locale == Locale.EN_IE:
            content = "A conversation between a user located in Ireland and a helpful assistant. The assistant is mindful of Irish spelling, vocabulary, entities and other Irish context."  # noqa: E501
        elif locale == Locale.EN_NZ:
            content = "A conversation between a user located in New Zealand and a helpful assistant. The assistant is mindful of New Zealand spelling, vocabulary, entities and other New Zealand context."  # noqa: E501
        elif locale == Locale.EN_ZA:
            content = "A conversation between a user located in South Africa and a helpful assistant. The assistant is mindful of South African spelling, vocabulary, entities and other South African context."  # noqa: E501
        elif locale == Locale.EN_IN:
            content = "A conversation between a user located in India and a helpful assistant. The assistant is mindful of Indian spelling, vocabulary, entities and other Indian context."  # noqa: E501
        elif locale == Locale.EN_SG:
            content = "A conversation between a user located in Singapore and a helpful assistant. The assistant is mindful of Singaporean spelling, vocabulary, entities and other Singaporean context."  # noqa: E501
        else:
            content = f"A conversation between a user and a helpful assistant. The user's locale is {locale.value}."
        return Message(role="system", content=content)

    @staticmethod
    def from_system(content: str) -> "Message":
        """
        Returns a new system message with the given content.

        Args:
            content (str): The text content.

        Returns:
            A new system message.
        """
        return {"role": "system", "content": content}

    @staticmethod
    def from_user(content: str) -> "Message":
        """
        Returns a new user message with the given content.

        Args:
            content (str): The text content.

        Returns:
            A new user message.
        """
        return {"role": "user", "content": content}

    @staticmethod
    def from_assistant(content: str) -> "Message":
        """
        Returns a new assistant message with the given content.

        Args:
            content (str): The text content.

        Returns:
            A assistant user message.
        """
        return {"role": "assistant", "content": content}


InstructMessages: TypeAlias = list[Message] | dict[list[Message]]
"""
A type representing an interaction with the language model.

For inference, this will contain the system and user message, for example:
```
[{"role": "system", "content": "<INSTRUCTION>"}, {"role": "user", "content": "<USER PROMPT>"}]
```

Alternatively, you can just pass in the user message which will prepend the default system message, for example:
```
[{"role": "user", "content": "<USER PROMPT>"}]
```

For supervised fine-tuning, each sample should include the system, user, and assistant message, for example:
```
[{"role": "system", "content": "<INSTRUCTION>"}, {"role": "user", "content": "<USER PROMPT>"}, {"role": "assistant", "content": "<ASSISTANT RESPONSE>"}]
```

Similar with inference, the default system message will be prepended if omitted. This can be extended to support multi-turn, for example:
```
[{"role": "user", "content": "<USER PROMPT>"}, {"role": "assistant", "content": "<ASSISTANT RESPONSE>"}, {"role": "user", "content": "<USER PROMPT>"}, {"role": "assistant", "content": "<ASSISTANT RESPONSE>"}, ...]
```

Note: Instruct messages can also be encapsulated in a dictionary, for example:
```
{"messages": [{"role": "user", "content": "<USER PROMPT>"}]}
```

"""  # noqa: E501
