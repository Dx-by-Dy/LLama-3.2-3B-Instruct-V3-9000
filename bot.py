import os

import telebot
from dotenv import load_dotenv

from chat import Chat
from support import BiTranslator, Model, Database
from user_config import UserConfig

load_dotenv()
TELEGRAM_API_KEY = os.getenv('TELEGRAM_API_KEY')

bot = telebot.TeleBot(TELEGRAM_API_KEY, threaded=False,
                      skip_pending=True, num_threads=1)
model = Model()
translator = BiTranslator()
users_chats: dict[int, UserConfig] = {}
database = Database()


def before_start(message):
    bot.send_message(message.from_user.id, "Напишите /start чтобы начать")


@bot.message_handler(commands=['start'])
def start_dialog(message):
    users_chats[message.from_user.id] = UserConfig(translator)
    bot.send_message(message.from_user.id, Chat.START_CHAT_REPLIC)


@bot.message_handler(commands=['restart'])
def start_dialog(message):
    if message.from_user.id not in users_chats:
        before_start(message)
        return

    database.restart(message.from_user.id)
    users_chats[message.from_user.id].clear(translator)
    bot.send_message(message.from_user.id, "RESTARTED")


@bot.message_handler(commands=['debug'])
def start_dialog(message):
    if message.from_user.id not in users_chats:
        before_start(message)
        return

    users_chats[message.from_user.id].debug = not users_chats[message.from_user.id].debug
    bot.send_message(message.from_user.id,
                     f"DEBUG FLAG: {users_chats[message.from_user.id].debug}")


@bot.message_handler(func=lambda message: message.content_type == "text")
def user_message(message):
    if message.from_user.id not in users_chats:
        before_start(message)
        return

    user_config = users_chats[message.from_user.id]
    user_config.chat.write_user_message(message.text, translator)
    database.write_user_message(message.from_user.id, message.text)
    model_output = model.generate(user_config.chat.model_chat)
    model_message = user_config.chat.write_model_message(model_output, translator)
    database.write_model_message(message.from_user.id, model_message)

    last_user_message = user_config.chat.last_user_message()
    last_model_message = user_config.chat.last_model_message()

    if user_config.debug:
        debug_info = "\n\n" + "-" * 45 + " DEBUG " + "-" * 46 + \
            f"\nUser message for model:\n{last_user_message['content_model_lang']}\n\nModel original ans:\n{last_model_message['content_model_lang']}"
    else:
        debug_info = ""

    bot.send_message(message.from_user.id,
                     f"{last_model_message['content_user_lang']}" + debug_info)

    print(users_chats[message.from_user.id].chat)


if __name__ == "__main__":
    bot.polling(none_stop=True)
