from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from scripts.utils import Utils
from config.constants import Constants


class TrainModelController:
    # 创建训练模型
    def create_model(self):
        file_name = Constants.INTENT_DATASET_PREPROCESSED_FILE_NAME
        file_dir = Constants.DATA_FILE_DIR
        file_path = file_dir + file_name
        # 读取经过预处理的数据
        data = Utils.read_csv(file_path)
        labels = data[Constants.INTENT_LABEL]
        # 生成数据集的向量空间
        vectorizer = CountVectorizer()
        vector_data = vectorizer.fit_transform(data[Constants.TEXT_LABEL])
        # TF_IDF转换词频
        tf_transformer = TfidfTransformer(
            use_idf=True, sublinear_tf=True).fit(vector_data)
        tf_train_data = tf_transformer.transform(vector_data)
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
        # 创建分类模型
        model = MultinomialNB()  # 使用朴素贝叶斯模型
        # 模型训练
        model.fit(X_train, y_train)
        # 存储模型到文件
        Utils.save_serialize_data(
            model, Constants.MODELS_FILE_DIR + Constants.MODELS_FILE_NAME)
        # 存储向量空间到文件
        Utils.save_serialize_data(
            vectorizer, Constants.MODELS_FILE_DIR + Constants.VECTOR_FILE_NAME)
        print("Create model succeed!")

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

    # 根据意图回答问题
    def answer_question(self, intent):
        intent_menu = Utils.read_json(Constants.INTENT_LABEL_FILEPATH)
        return intent_menu[str(intent)]

    # 使用模型对输入内容进行意图预测
    def predict_intent(self, user_input):
        vectorizer = Utils.read_serialize_data(
            Constants.MODELS_FILE_DIR + Constants.VECTOR_FILE_NAME)
        model = Utils.read_serialize_data(
            Constants.MODELS_FILE_DIR + Constants.MODELS_FILE_NAME)
        input_vector = vectorizer.transform([user_input])
        intent_num = model.predict(input_vector)[0]
        # print("Intent Number:" + str(intent))
        return self.answer_question(intent_num)

    def chat(self):
        print("Welcome to the chatbot! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            filtered_input = Utils.clean_input(user_input)
            if filtered_input == None:
                continue
            if filtered_input.lower() == "exit":
                print("Goodbye!")
                break
            response = self.predict_intent(filtered_input)
            print("Bot:", response)
