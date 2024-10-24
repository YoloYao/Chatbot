import nltk
import string
from urllib import request
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from scripts.utils import Utils


class PreProcessController:
    # 分词
    def tokenize(self, input):
        tokens = nltk.word_tokenize(input)
        return tokens

    # 去除标点符号
    def remove(self, file_path):
        string.punctuation = string.punctuation + "’" + "-" + "‘" + "-"
        string.punctuation = string.punctuation.replace(".", "")
        file = open(file_path, encoding="utf8").read()
        file_nl_removed = ""
        for line in file:
            # removes newline characters
            line_nl_removed = line.replace("\n", " ")
            file_nl_removed += line_nl_removed  # adds filtered line to the list
            # joins all the lines in the list in a single string
            file_p = "".join(
                [char for char in file_nl_removed if char not in string.punctuation])
        return file_p

    # 下载语料库
    def download_corpus(self, corpus_name):
        nltk.download(corpus_name)
        selected_file = 'stopwords.txt'
        text_content = stopwords.words('english')
        Utils.write_file_list(selected_file, text_content)
        print("finish")

    # 下载网页信息
    def download_web_info(self, url):
        # url = "http://example.org"
        content = request.urlopen(url).read().decode('utf-8', errors='ignore')
        return content
