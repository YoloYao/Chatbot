#!/usr/bin/env python3

from scripts.utils import Utils
from scripts.preprocess import PreProcessController
from scripts.train import TrainModelController
import traceback


class Chatbot:
    def init(self):
        self.name = "YoYo"
        self.pre_processor = PreProcessController()
        self.model_trainer = TrainModelController()

    def run(self):
        Utils.connectSSL()
        # print("My name is " + self.name)
        # 1.预处理数据
        # self.pre_processor.preprocess_csv_corpus()
        # self.pre_processor.preprocess_corpus()
        # 2.生成模型
        # self.model_trainer.create_model()
        # 3.基于意图识别进行聊天
        self.model_trainer.chat()


if __name__ == '__main__':
    try:
        chatbot = Chatbot()
        chatbot.init()
        chatbot.run()
    except Exception as e:
        print("error:{e}")
        traceback.print_exc()
