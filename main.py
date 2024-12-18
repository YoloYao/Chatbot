#!/usr/bin/env python3

from scripts.utils import Utils
from scripts.preprocess import PreProcessController
from scripts.train import TrainModelController
from scripts.chat import ChatController
import traceback


class Chatbot:
    def init(self):
        self.pre_processor = PreProcessController()
        self.model_trainer = TrainModelController()
        self.chat_controller = ChatController()

    def run(self):
        Utils.connectSSL()
        # 1.预处理数据 preprocess data
        # self.pre_processor.refresh_dataset_numbers()
        # self.pre_processor.preprocess_corpus()
        # 2.生成模型 generate model and vector space
        # self.model_trainer.create_model()
        # 3.基于意图识别进行聊天 Chat based on intention recognition
        self.chat_controller.chat()


if __name__ == '__main__':
    try:
        Utils.clear_screen()
        chatbot = Chatbot()
        chatbot.init()
        chatbot.run()
    except Exception as e:
        print("error:{e}")
        traceback.print_exc()
