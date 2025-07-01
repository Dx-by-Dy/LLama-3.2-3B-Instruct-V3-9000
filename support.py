import argostranslate.package
import argostranslate.translate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

BEGIN_TOKEN: str = "<|begin_of_text|>"
START_PROMT: str = "You are a coach assistant who must help the user solve his problems using your knowledge. Be calm like a professional, don't show emotions. First of all, ask what advice the user came to your chat for."
EOT_TOKEN: str = "<|eot_id|>"
START_HEADER_TOKEN: str = "<|start_header_id|>"
END_HEADER_TOKEN: str = "<|end_header_id|>\n\n"
SYSTEM_NAME: str = "system"
ASSISTANT_NAME: str = "assistant"
USER_NAME: str = "user"
SYSTEM_START_PROMT: str = BEGIN_TOKEN + START_HEADER_TOKEN + \
    SYSTEM_NAME + END_HEADER_TOKEN + START_PROMT + EOT_TOKEN

MODEL_PATH: str = "model/checkpoint-9000"


class Model:
    def __init__(self):
        self.lang = "en"
        self.max_new_tokens = 512
        self.model = load_model()
        self.model.eval()
        self.tokenizer = load_tokenizer()

    def generate(self, chat: str) -> str:
        inputs = self.tokenizer(chat, return_tensors="pt").to("cuda")
        return self.tokenizer.batch_decode(
            self.model.generate(**inputs, max_new_tokens=self.max_new_tokens))[0]


# class BiTranslator:
#     def __init__(self, from_lang: str, to_lang: str):
#         self.forward_translator, self.backward_translator = load_translators(
#             from_lang, to_lang)

#     def forward(self, message: str) -> str:
#         return self.forward_translator.translate(message)

#     def backward(self, message: str) -> str:
#         return self.backward_translator.translate(message)


def load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
    )


# def load_translators(from_lang: str, to_lang: str):
#     argostranslate.package.update_package_index()
#     available_packages = argostranslate.package.get_available_packages()
#     available_package = list(
#         filter(
#             lambda x: x.from_code == from_lang and x.to_code == to_lang, available_packages
#         )
#     )[0]
#     download_path = available_package.download()
#     argostranslate.package.install_from_path(download_path)

#     available_package = list(
#         filter(
#             lambda x: x.from_code == to_lang and x.to_code == from_lang, available_packages
#         )
#     )[0]
#     download_path = available_package.download()
#     argostranslate.package.install_from_path(download_path)

#     installed_languages = argostranslate.translate.get_installed_languages()

#     from_to_translator = list(filter(
#         lambda x: x.code == from_lang,
#         installed_languages))[0].get_translation(list(filter(
#             lambda x: x.code == to_lang,
#             installed_languages))[0])

#     to_from_translator = list(filter(
#         lambda x: x.code == to_lang,
#         installed_languages))[0].get_translation(list(filter(
#             lambda x: x.code == from_lang,
#             installed_languages))[0])

#     return (from_to_translator, to_from_translator)

class BiTranslator:
    def __init__(self):
        self.model, self.tokenizer = load_translators()
        self.forward_prefix = "translate to ru: "
        self.backward_prefix = "translate to en: "

    def forward(self, message: str) -> str:
        if message.replace(" ", "") == "":
            return "..."
        input_ids = self.tokenizer(self.forward_prefix + message, return_tensors="pt").to("cuda")
        return self.tokenizer.batch_decode(self.model.generate(**input_ids, max_new_tokens=512), skip_special_tokens=True)[0]

    def backward(self, message: str) -> str:
        input_ids = self.tokenizer(self.backward_prefix + message, return_tensors="pt").to("cuda")
        return self.tokenizer.batch_decode(self.model.generate(**input_ids, max_new_tokens=512), skip_special_tokens=True)[0]

def load_translators():
    model_name = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
    return T5ForConditionalGeneration.from_pretrained(model_name).to("cuda"), T5Tokenizer.from_pretrained(model_name)
