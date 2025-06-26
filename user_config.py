from chat import Chat
from support import BiTranslator


class UserConfig:
    def __init__(self, translator: BiTranslator):
        self.chat = Chat(translator)
        self.debug = False

    def clear(self, translator: BiTranslator):
        self.chat = Chat(translator)
