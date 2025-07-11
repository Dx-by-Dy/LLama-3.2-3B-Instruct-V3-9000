import argostranslate.package
import argostranslate.translate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import sqlite3
import datetime

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

class Database:
    def __init__(self):
        self.conn = sqlite3.connect("data.db")
        self.cur = self.conn.cursor()

        self.cur.execute("CREATE TABLE IF NOT EXISTS messages(id PRIMAL KEY, user_id integer, message TEXT, time integer, role integer)")
    
    def write_user_message(self, user_id: int, message: str) -> None:
        now = int(datetime.datetime.now().timestamp())
        self.cur.executemany("INSERT INTO messages(user_id, message, time, role) values (?, ?, ?, ?)", [(user_id, message, now, 1)])
        self.conn.commit()

    def write_model_message(self, user_id: int, message: str) -> None:
        now = int(datetime.datetime.now().timestamp())
        self.cur.executemany("INSERT INTO messages(user_id, message, time, role) values (?, ?, ?, ?)", [(user_id, message, now, 2)])
        self.conn.commit()

    def restart(self, user_id) -> None:
        now = int(datetime.datetime.now().timestamp())
        self.cur.executemany("INSERT INTO messages(user_id, message, time, role) values (?, ?, ?, ?)", [(user_id, "RESTARTED", now, 0)])
        self.conn.commit()