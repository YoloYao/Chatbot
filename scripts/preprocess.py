import nltk
import string
import re
import pandas as pd
import copy
from urllib import request
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
from nltk import ngrams
from scripts.utils import Utils
from collections import Counter
# from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.text import Tokenizer


class PreProcessController:
    def __init__(self):
        Utils.connectSSL()
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        nltk.download('wordnet')
        nltk.download('punkt')

    # 分词
    def tokenize(self, input):
        tokens = nltk.word_tokenize(input)
        return tokens

    # 去除标点符号
    def clean_text(self, content):
        content = re.sub(r'<.*?>', '', content)
        string.punctuation = string.punctuation + "’" + "-" + "‘" + "-"
        string.punctuation = string.punctuation.replace(".", "")
        content_nl_removed = ""
        for line in content:
            # removes newline characters
            line_nl_removed = line.replace("\n", " ")
            content_nl_removed += line_nl_removed  # adds filtered line to the list
            # joins all the lines in the list in a single string
            content_p = "".join(
                [char for char in content_nl_removed if char not in string.punctuation])
        # 转换为小写
        content_p = content_p.lower().strip()
        return content_p

    # 去除停用词
    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    # 词干还原
    def lemmatize(self, tokens):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    # 拼写检查
    def correct_spelling(self, tokens):
        spell = SpellChecker()
        corrected_tokens = [spell.correction(token) for token in tokens]
        return corrected_tokens

    # 生成n-gram特征
    def generate_ngrams(self, tokens, n):
        n_grams = list(ngrams(tokens, n))
        return [' '.join(grams) for grams in n_grams]
    
    # 预处理单个文本数据
    def preprocess_data(self, input_data):
        content = self.clean_text(input_data)
        # 2.分词（Tokenization）
        tokens = self.tokenize(content)
        # 3.去除停用词（Stopword Removal）
        tokens = self.remove_stopwords(tokens)
        # 4.词干提取或词形还原（Stemming/Lemmatization）
        tokens = self.lemmatize(tokens)
        # 5.移除少见和高频词
        # tokens = self.remove_rare_and_frequent_words(tokens)
        # 6.拼写检查
        # tokens = self.correct_spelling(tokens)
        # 7.生成ngram特征
        # tokens = self.generate_ngrams(tokens, 2)
        return tokens

    # 预处理文本数组数据
    def preprocess_list_data(self, list_data):
        list = copy.deepcopy(list_data)
        for i in range(len(list)):
            list[i] = " ".join(self.preprocess_data(list[i]))
        return list

    # 生成测试集
    def gen_test_dataset(self, data, file_dir, file_name):
        labels = data['intent']
        vectorizer = CountVectorizer()
        vector_data = vectorizer.fit_transform(data['text'])
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(
            vector_data, labels)
        train_data = {'text': [], 'label': []}
        val_data = {'text': [], 'label': []}
        test_data = {'text': [], 'label': []}
        train_data['text'].append(X_train)
        train_data['label'].append(y_train)
        val_data['text'].append(X_val)
        val_data['label'].append(y_val)
        test_data['text'].append(X_test)
        test_data['label'].append(y_test)
        Utils.save_padding_data(
            (X_train, y_train), file_dir + 'train/train_' + Utils.change_file_suffix(file_name))
        Utils.save_padding_data(
            (X_val, y_val), file_dir + 'val/val_' + Utils.change_file_suffix(file_name))
        Utils.save_padding_data(
            (X_test, y_test), file_dir + 'test/test_' + Utils.change_file_suffix(file_name))
        model = MultinomialNB()  # 使用朴素贝叶斯模型
        model.fit(X_train, y_train)
        # save model
        Utils.save_padding_data(
            model, file_dir + 'model_' + Utils.change_file_suffix(file_name))
        # save vector space
        Utils.save_padding_data(
            vectorizer, file_dir + 'vector_' + Utils.change_file_suffix(file_name))
        print("Split dataset succeeded!")

    # 向量化
    def vectorize(self, texts):
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(texts)  # 将文本转换为特征向量

    # 数据集拆分
    def split_dataset(self, X, y):
        test_size = 0.2
        validation_size = 0.1
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size)
        return X_train, X_val, X_test, y_train, y_val, y_test

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

    # 聚类生成意图标签
    def create_labels(self, texts, num_clusters):
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=num_clusters, random_state=50)  # 假设我们希望分5类
        kmeans.fit(X)
        labels = kmeans.labels_
        return labels

    # 预处理csv数据集
    def preprocess_csv_corpus(self):
        file_name = 'intent_dataset.csv'
        file_dir = './data/'
        file_path = file_dir + file_name
        data = pd.read_csv(file_path)
        tokens = self.preprocess_list_data(data['text'])
        # 添加标签
        # labels = self.create_labels(tokens2, 5)
        data['cleaned text'] = tokens
        # for i in range(5):
        #     print(f"\nCluster {i} examples:")
        #     print(data['intent'][i])
        #     print(tokens2[data['intent'] == i].head())
        
        data.to_csv("./data/labeled_intent_dataset.csv", index=False)
        self.gen_test_dataset(data, file_dir, file_name)
        print("Preprocess data succeeded!")

    # 预处理文本数据集
    def preprocess_corpus(self):
        file_name = 'dataset1.txt'
        file_dir = './data/'
        file_path = file_dir + file_name
        preprocessed_data = {
            "text": [],
            "token": [],
            "label": []
        }
        content = Utils.read_file(file_path)
        sentences = sent_tokenize(content)
        preprocessed_data["text"] = copy.deepcopy(sentences)
        tokens = self.preprocess_list_data(sentences)
        preprocessed_data["token"] = tokens
        # 添加标签
        labels = self.create_labels(tokens, 5)
        preprocessed_data["label"] = labels
        data_frame = pd.DataFrame(preprocessed_data)
        # 向量化数据
        vector_data = self.vectorize(tokens)
        # 划分训练集、测试集
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(
            vector_data, labels)
        # 存入文件
        data_frame.to_csv("./data/labeled_dataset1.csv", index=False)
        # 7.数据拆分
        # 8.保存预处理数据
        print("Preprocess data succeeded!")
