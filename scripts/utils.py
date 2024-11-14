import ssl
import pickle
import json
import re
import os
import pandas as pd
from nltk.corpus.reader.util import StreamBackedCorpusView


class Utils:
    # mac系统需要执行SSL连接方法连接到nltk工具库
    @staticmethod
    def connectSSL():
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

    # 在终端窗口清屏
    @staticmethod
    def clear_screen():
        # 如果是 Windows 系统
        if os.name == 'nt':
            os.system('cls')
        # 如果是其他系统 (如 Linux、macOS)
        else:
            os.system('clear')
    
    # 读取JSON文件内容
    @staticmethod
    def read_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            intent_labels = json.load(f)
        return intent_labels

    # 读取本地停用词文件
    @staticmethod
    def load_stopwords(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            stopwords = set(line.strip() for line in file)
        return stopwords

    # 修改文件后缀
    @staticmethod
    def change_file_suffix(file_name):
        return file_name.replace('.txt', '.pkl').replace('.csv', '.pkl')

    # 读取csv文件内容
    @staticmethod
    def read_csv(file_path):
        return pd.read_csv(file_path)

    # 读取文件内容
    @staticmethod
    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()  # 一次性读取整个文件
        return content
    
    # 逐行读取文件内容并返回数组
    @staticmethod
    def read_file_by_line(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().splitlines()  # 使用 splitlines() 自动去除换行符
        return lines

    # 写入文件内容（覆盖）
    @staticmethod
    def write_file_cover(file_path, content):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

    # 写入文件内容（追加）
    @staticmethod
    def write_file(file_path, content):
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(content)

    # 写入文件列表内容（追加）
    @staticmethod
    def write_file_list(file_path, content):
        with open(file_path, 'a', encoding='utf-8') as file:
            for word in content:
                file.write(word + '\n')

    # 写入语料库内容到文件
    @staticmethod
    def write_corpus(file_path, out_file_path):
        corpus_view = StreamBackedCorpusView(
            file_path, read_block=lambda stream: stream.readlines())
        # 打开一个文本文件用于写入
        with open(out_file_path, 'w', encoding='utf-8') as output_file:
            # 逐行从语料库读取并写入到目标文件
            for block in corpus_view:
                output_file.writelines(block)  # 写入到文件

    # 序列化内容存储到文件
    @staticmethod
    def save_serialize_data(data, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    # 读取序列化文件
    @staticmethod
    def read_serialize_data(file_path):
        content = None
        with open(file_path, 'rb') as file:
            content = pickle.load(file)
        return content

    # 清理用户输入，去除空字符串、换行符等无效内容
    @staticmethod
    def clean_input(user_input):
        # 去除前后空格和换行符
        cleaned_input = user_input.strip()
        # 去除所有多余的空格和换行符
        cleaned_input = re.sub(r'\s+', ' ', cleaned_input)

        # 检查清理后的字符串是否为空或仅包含空白字符
        if not cleaned_input or cleaned_input == "":
            return None  # 无效输入
        return cleaned_input
