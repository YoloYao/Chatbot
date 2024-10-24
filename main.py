#!/usr/bin/env python3

from scripts.utils import Utils
from scripts.preprocess import PreProcessController
import nltk
import traceback


class Chatbot:
    def init(self):
        self.name = "YoYo"
        self.pre_processor = PreProcessController()

    def run(self):
        Utils.connectSSL()
        print("My name is " + self.name)
        self.pre_processor.download_corpus('stopwords')


if __name__ == '__main__':
    try:
        chatbot = Chatbot()
        chatbot.init()
        chatbot.run()
    except Exception as e:
        print("error:{e}")
        traceback.print_exc()
