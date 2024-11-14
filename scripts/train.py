from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from scripts.utils import Utils
from config.constants import Constants
from models.model import Model
import pandas as pd


class TrainModelController:
    # 创建训练模型
    def create_model(self):
        # 生成问题模型
        file_name1 = Constants.INTENT_DATASET_PREPROCESSED_INDEX + \
            Constants.INTENT_DATASET_FILE_NAME
        label1 = Constants.QUESTION_LABEL
        self.generate_model(file_name1, label1, need_split=True)
        self.generate_intent_model(file_name1, label1)
        # 生成答案模型
        file_name2 = Constants.INTENT_DATASET_PREPROCESSED_INDEX + \
            Constants.INTENT_ANSWER_FILE_NAME
        label2 = Constants.ANSWER_LABEL
        self.generate_model(file_name2, label2, need_split=False)
        self.generate_intent_model(file_name2, label2)

    # 根基不同意图划分数据并生成模型集合
    def generate_intent_model(self, file_name, label_name):
        file_dir = Constants.DATA_FILE_DIR
        file_path = file_dir + file_name
        # 读取经过预处理的数据
        data = Utils.read_csv(file_path)
        df = pd.DataFrame(data)
        intent_menu = Utils.read_json(Constants.INTENT_LABEL_FILEPATH)
        trans_models = []
        for i in range(Constants.TOTAL_INTENT_NUM):
            # 过滤出对应intent的所有answer
            filtered_data = df[df[Constants.INTENT_LABEL] ==
                               i][label_name].reset_index(drop=True)
            cleaned_data = df[df[Constants.INTENT_LABEL] ==
                              i][Constants.CLEANED_LABEL].reset_index(drop=True)
            new_labels = df[df[Constants.INTENT_LABEL] ==
                            i][Constants.INTENT_LABEL].reset_index(drop=True)
            numbers = df[df[Constants.INTENT_LABEL] ==
                            i][Constants.NUMBER_LABEL].reset_index(drop=True)
            transModel = Model()
            transModel.init(None, None)
            # 生成数据集的向量空间
            answer_vec_data = transModel.vectorizer.fit_transform(
                cleaned_data.astype(str))

            # 使用 TF-IDF 转换词频矩阵
            # TF_IDF转换词频
            transModel.tf_data = transModel.tf_transformer.fit_transform(
                answer_vec_data)

            transModel.data = filtered_data
            trans_models.append(transModel)
            # 存储原数据到文件
            csv_data = {
                Constants.INTENT_LABEL: [],
                Constants.ANSWER_LABEL: [],
                Constants.NUMBER_LABEL: []
            }
            data_file_name = Constants.MODELS_FILE_DIR + label_name + '/' + \
                label_name+'_' + \
                intent_menu[str(i)]+Constants.DATA_CSV_FILE_NAME
            csv_data[Constants.INTENT_LABEL] = new_labels
            csv_data[Constants.ANSWER_LABEL] = filtered_data
            csv_data[Constants.NUMBER_LABEL] = numbers
            data_frame = pd.DataFrame(csv_data)
            data_frame.to_csv(data_file_name, encoding='utf-8', index=False)

        trans_model_file_name = Constants.MODELS_FILE_DIR + \
            label_name + Constants.TRANS_MODEL_FILE_PATH
        Utils.save_serialize_data(
            trans_models, trans_model_file_name)
        print(f"Split {label_name} intent data succeed!")

    # 生成模型
    def generate_model(self, file_name, label_name, need_split):
        file_dir = Constants.DATA_FILE_DIR
        file_path = file_dir + file_name
        # 读取经过预处理的数据
        data = Utils.read_csv(file_path)
        labels = data[Constants.INTENT_LABEL]
        # 生成数据集的向量空间
        vectorizer = CountVectorizer()
        vector_data = vectorizer.fit_transform(
            data[Constants.CLEANED_LABEL].astype(str))
        # 使用 TF-IDF 转换词频矩阵
        # TF_IDF转换词频
        tf_transformer = TfidfTransformer(
            use_idf=True, sublinear_tf=True).fit(vector_data)
        tf_train_data = tf_transformer.transform(vector_data)
        # 创建分类模型
        model = MultinomialNB()  # 使用朴素贝叶斯模型
        # 模型训练
        if need_split == True:
            X_train, y_train = self.gen_test_dataset(
                tf_train_data, labels, file_dir, file_name)
            model.fit(X_train, y_train)
        else:
            model.fit(tf_train_data, labels)
        # 存储模型到文件
        model_file_name = Constants.MODELS_FILE_DIR + \
            label_name + Constants.MODELS_FILE_NAME
        Utils.save_serialize_data(
            model, model_file_name)
        # 存储向量空间到文件
        vector_file_name = Constants.MODELS_FILE_DIR + \
            label_name+Constants.VECTOR_FILE_NAME
        Utils.save_serialize_data(
            vectorizer, vector_file_name)
        print(f"Create {label_name} model succeed!")

    def gen_test_dataset(self, tf_train_data, labels, file_dir, file_name):
        # 划分训练集
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(
            tf_train_data, labels)
        # 训练集写入文件
        Utils.save_serialize_data(
            (X_train, y_train), file_dir + 'train/train_' + Utils.change_file_suffix(file_name))
        Utils.save_serialize_data(
            (X_val, y_val), file_dir + 'validation/validation_' + Utils.change_file_suffix(file_name))
        Utils.save_serialize_data(
            (X_test, y_test), file_dir + 'test/test_' + Utils.change_file_suffix(file_name))
        return X_train, y_train

    # 数据集拆分
    def split_dataset(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Constants.SPLIT_TEST_SIZE, random_state=Constants.RANDOM_RATE_1)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=Constants.SPLIT_VALIDATION_SIZE, random_state=Constants.RANDOM_RATE_1)
        return X_train, X_val, X_test, y_train, y_val, y_test

    # 模型训练
    def train(self, X_train, y_train):
        model = MultinomialNB()  # 使用朴素贝叶斯模型
        model.fit(X_train, y_train)
