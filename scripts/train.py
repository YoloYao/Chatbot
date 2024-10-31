import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from scripts.utils import Utils


class TrainModelController:
    def load_preprocessed_data(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    # 模型训练
    def train(self, X_train, y_train):
        model = MultinomialNB()  # 使用朴素贝叶斯模型
        model.fit(X_train, y_train)

    # 使用模型进行预测
    def predict_intent(self, user_input, model):
        vectorizer = Utils.read_padding_data(
            './data/vector_intent_dataset.pkl')
        input_vector = vectorizer.transform([user_input])
        intent = model.predict(input_vector)[0]
        return intent

    def answer_question(self, intent):
        intent_menu = Utils.read_json('./data/intent_labels.json')
        print(intent_menu[str(intent)])
        
    def predict_test(self):
        (X_train, y_train) = self.load_preprocessed_data(
            './data/train/train_intent_dataset.pkl')
        model = MultinomialNB()  # 使用朴素贝叶斯模型
        model.fit(X_train, y_train)
        user_input = "I want to change my password"
        intent = self.predict_intent(user_input, model)
        print("Intent:" + str(intent))
        self.answer_question(intent)
