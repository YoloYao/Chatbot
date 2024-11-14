import nltk
import string
import re
import pandas as pd
import copy
from urllib import request
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk import ngrams
from scripts.utils import Utils
from config.constants import Constants
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class PreProcessController:
    def init(self):
        Utils.connectSSL()
        # nltk.download('stopwords', quiet=True)
        # self.stop_words = set(stopwords.words('english'))
        self.stop_words = Utils.load_stopwords(Constants.STOPWORDS_FILEPATH)

    # 去除标点符号
    def clean_text(self, content):
        content = re.sub(r'<.*?>', '', content)
        string.punctuation = string.punctuation + "’" + "-" + "‘" + "-"
        # 不去除句号
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

    # 分词
    def tokenize(self, input):
        tokens = nltk.word_tokenize(input)
        return tokens

    # 去除停用词
    def remove_stopwords(self, tokens):
        # 特殊停用词检查
        check_result = self.check_stopwords(" ".join(tokens))
        if check_result == True:
            return tokens
        return [word for word in tokens if word not in self.stop_words]

    # 词干还原
    def lemmatize(self, tokens):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    # 检查停用词处理是否跳过
    def check_stopwords(self, check_str):
        special_strs = Utils.read_file_by_line(
            Constants.SPECIAL_SENTENCES_FILEPATH)
        for item in special_strs:
            if item in check_str:
                return True
        return False

    # 生成n-gram特征
    def generate_ngrams(self, tokens, n):
        n_grams = list(ngrams(tokens, n))
        return [' '.join(grams) for grams in n_grams]

    # 预处理单个文本数据
    def preprocess_data(self, input_data):
        # 1.清洗数据（Clean Data）
        content = self.clean_text(input_data)
        # 2.分词（Tokenization）
        tokens = self.tokenize(content)
        # 3.词干提取或词形还原（Stemming/Lemmatization）
        tokens = self.lemmatize(tokens)
        # 4.去除停用词（Stopword Removal）
        tokens = self.remove_stopwords(tokens)
        return tokens

    # 预处理文本数组数据
    def preprocess_list_data(self, list_data):
        list = copy.deepcopy(list_data)
        for i in range(len(list)):
            list[i] = " ".join(self.preprocess_data(list[i]))
        return list

    # 下载语料库
    def download_corpus(self, corpus_name, aim_file_name):
        nltk.download(corpus_name)
        text_content = stopwords.words('english')
        Utils.write_file_list(aim_file_name, text_content)
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
    def preprocess_corpus(self):
        # 预处理问题数据集
        file_name = Constants.INTENT_DATASET_FILE_NAME
        label_name = Constants.QUESTION_LABEL
        self.preprocess_csv_corpus(file_name, label_name)
        # 预处理答案数据集
        file_name1 = Constants.INTENT_ANSWER_FILE_NAME
        label_name1 = Constants.ANSWER_LABEL
        self.preprocess_csv_corpus(file_name1, label_name1)

    def preprocess_csv_corpus(self, file_name, label_name):
        self.init()
        file_dir = Constants.DATA_FILE_DIR
        file_path = file_dir + file_name
        data = Utils.read_csv(file_path)
        # 预处理数据集中的数据
        sentences = self.preprocess_list_data(data[label_name])
        # 添加标签(根据数据集内容进行聚类，区分出不同意图类别)
        data['cleaned text'] = sentences
        # 预处理后的数据存入文件
        data.to_csv(
            file_dir + Constants.INTENT_DATASET_PREPROCESSED_INDEX + file_name, index=False)
        print(f"Preprocess {label_name} data succeed!")

    # 给数据集分类生成标签
    def generate_labeled_csv_corpus(self):
        self.init()
        file_path = './data/CW1_Dataset.csv'
        csv_data = Utils.read_csv("data/CW1-Dataset.csv")
        questions = csv_data['Question']
        sentences = self.preprocess_list_data(questions)
        labels = self.create_labels(sentences, 5)
        # for i in range(5):
        #     print(f"\nCluster {i} examples:")
        #     print(data[Constants.INTENT_LABEL][i])
        #     print(tokens2[data[Constants.INTENT_LABEL] == i].head())
        csv_data['Label'] = labels
        csv_data.to_csv('./data/labeled_CW1_Dataset.csv', index=False)

    def refresh_dataset_numbers(self):
        self.init()
        csv_data1 = pd.read_csv("data/dataset2.csv", encoding='latin1')
        csv_data2 = pd.read_csv("data/answer2.csv", encoding='latin1')
        num_list1 = []
        for i in range(len(csv_data1['intent'])):
            num_list1.append(i)
            i += 1
        csv_data1['number'] = num_list1
        num_list2 = []
        for i in range(len(csv_data2['intent'])):
            num_list2.append(i)
            i += 1
        csv_data2['number'] = num_list2
        # new_data = {
        #     "intent": [],
        #     "question": []
        # }
        # answer_data = {
        #     "intent": [],
        #     "answer": []
        # }
        # new_data['intent'] = intents
        # new_data['question'] = questions
        # answer_data['intent'] = intents
        # answer_data['answer'] = answers
        # data_frame1 = pd.DataFrame(new_data)
        # data_frame2 = pd.DataFrame(answer_data)
        # data_frame1.to_csv('./data/small_talk_dataset.csv', index=False)
        # data_frame2.to_csv('./data/answer/small_talk_answer.csv', index=False)
        csv_data1.to_csv('./data/dataset.csv', encoding='utf-8', index=False)
        csv_data2.to_csv('./data/answer.csv', encoding='utf-8', index=False)

    # 预处理文本数据集
    def preprocess_text_corpus(self):
        self.init()
        file_name = 'dataset1.txt'
        file_dir = Constants.DATA_FILE_DIR
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
