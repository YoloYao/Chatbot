# 导入所需库
import pickle
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

user_identity = {}
# 假设我们有一个预处理好的数据集 'preprocessed_data.csv'
# 数据集包含两列："text" 和 "label"，分别表示用户输入和对应的意图
# try:
#     df = pd.read_csv("./data/intent_dataset.csv")
# except UnicodeDecodeError as e:
#     print("文件编码错误信息:", e)
data = pd.read_csv('./data/intent_dataset.csv', encoding='utf-8')

# 数据分离
texts = data['text']  # 用户输入的文本
labels = data['intent']  # 目标意图

# 向量化处理文本数据
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)  # 将文本转换为特征向量
y = labels

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 模型训练
model = MultinomialNB()  # 使用朴素贝叶斯模型
model.fit(X_train, y_train)

# 模型预测与评估
y_pred = model.predict(X_test)
print(f"模型准确率: {accuracy_score(y_test, y_pred):.2f}")
print("分类报告:\n", classification_report(y_test, y_pred))

# 保存模型和向量器
with open('intent_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("模型和向量器已保存。")

# 加载模型和向量器
with open('intent_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# 使用模型进行预测
def predict_intent(user_input):
    input_vector = vectorizer.transform([user_input])
    intent = model.predict(input_vector)[0]
    return intent


# def handle_intent(intent, user_input):
#     if intent == "booking":
#         return "Sure, I can help with booking. Where would you like to go?"
#     elif intent == "identity":
#         name = user_input.split()[-1]  # 假设用户输入 "My name is X"
#         user_identity['name'] = name
#         return f"Hello, {name}! How can I assist you today?"
#     elif intent == "transaction":
#         return "Please provide more details for your order."
#     elif intent == "question":
#         return "Let me check the information for you."
#     elif intent == "small_talk":
#         return random.choice(["Hello!", "How can I help?", "Good to see you!"])
#     else:
#         return "I'm not sure I understand. Could you please rephrase?"

def handle_intent(intent, user_input):
    if intent == 1:
        return f"Hello! How can I assist you today?"
    elif intent == 2:
        return "Please provide more details for your order."
    elif intent == "question":
        return "Let me check the information for you."
    elif intent == 0:
        return random.choice(["Hello!", "How can I help?", "Good to see you!"])
    else:
        return "I'm not sure I understand. Could you please rephrase?"
# 聊天交互


def chat():
    print("Welcome to the chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        intent = predict_intent(user_input)
        response = handle_intent(intent, user_input)
        print("Bot:", response)


# 启动聊天
# chat()
# 示例：对用户输入进行意图预测
user_input = "I want to change my password"
intent = predict_intent(user_input)
print("Predicted intent:", intent)
