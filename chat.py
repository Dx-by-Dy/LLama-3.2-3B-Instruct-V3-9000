from support import (ASSISTANT_NAME,
                     END_HEADER_TOKEN,
                     EOT_TOKEN,
                     START_HEADER_TOKEN,
                     START_PROMT,
                     SYSTEM_NAME,
                     SYSTEM_START_PROMT,
                     USER_NAME,
                     BiTranslator)
import re


class Chat:
    START_CHAT_REPLIC: str = "Привет, я коуч-бот, который поможет тебе решить твои проблемы! Напиши что тебя беспокоит."

    def __init__(self, translator: BiTranslator):
        self.model_chat = SYSTEM_START_PROMT
        self.chat: list[dict[str, str]] = [
            {"role": SYSTEM_NAME,
             "content_model_lang": START_PROMT,
             "content_user_lang": translator.forward(START_PROMT)}]

    def write_model_message(self, message: str, translator: BiTranslator) -> None:
        content = message.replace("\n", "").rsplit("{}".format(END_HEADER_TOKEN.replace('\n', '')), 1)[
            1].replace(EOT_TOKEN, "")
        content = re.sub(r"<\|.*?\|>", "", content)
        translated_content = translator.forward(content)
        self.chat.append({"role": ASSISTANT_NAME, "content_model_lang": content,
                         "content_user_lang": translated_content})
        self.model_chat += START_HEADER_TOKEN + \
            ASSISTANT_NAME + END_HEADER_TOKEN + \
            content

    def write_user_message(self, message: str, translator: BiTranslator) -> None:
        translated_message = translator.backward(message)
        self.chat.append(
            {"role": USER_NAME, "content_model_lang": translated_message, "content_user_lang": message})
        self.model_chat += START_HEADER_TOKEN + \
            USER_NAME + END_HEADER_TOKEN + \
            translated_message + EOT_TOKEN

    def get_assistant_replic(self) -> None:
        inputs = self.tokenizer(
            self.model_chat, return_tensors="pt").to("cuda")
        self.add_assistant_message(self.tokenizer.batch_decode(
            self.model.generate(**inputs, max_new_tokens=512))[0])

    def last_user_message(self) -> dict[str, str] | None:
        for idx in range(len(self.chat) - 1, -1, -1):
            if self.chat[idx]["role"] == USER_NAME:
                return self.chat[idx]
        return None

    def last_model_message(self) -> dict[str, str] | None:
        for idx in range(len(self.chat) - 1, -1, -1):
            if self.chat[idx]["role"] == ASSISTANT_NAME:
                return self.chat[idx]
        return None

    def __repr__(self):
        res = ""
        for replic in self.chat:
            res += f"{replic['role']}:\n\t{replic['content_user_lang']}\n\t{replic['content_model_lang']}\n"
        return res
