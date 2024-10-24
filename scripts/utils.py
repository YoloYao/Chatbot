import ssl
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

    # 读取文件内容
    @staticmethod
    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()  # 一次性读取整个文件
        return content

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
