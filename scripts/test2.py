# 导入必要的库
import re
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk

# 下载 nltk 停用词
nltk.download('stopwords')

# 停用词列表
stop_words = set(stopwords.words('english'))

# 示例数据集
# data = [
#     ("I want to book a flight", "booking"),
#     ("Book a flight to New York", "booking"),
#     ("My name is John", "identity"),
#     ("Order food", "transaction"),
#     ("What's the weather like?", "question"),
#     ("Hello", "small_talk")
# ]
content = pd.read_csv('./data/intent_dataset.csv')
data = content['text']
# 分离数据和标签
texts, labels = zip(*data)

# 数据预处理
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stop_words])

# 处理后的数据
processed_texts = [preprocess_text(text) for text in texts]

# 向量化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_texts)
y = labels

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = MultinomialNB()
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 对话系统的实现
user_identity = {}

def identify_intent(user_input):
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    intent = model.predict(input_vector)[0]
    return intent

def handle_intent(intent, user_input):
    if intent == "booking":
        return "Sure, I can help with booking. Where would you like to go?"
    elif intent == "identity":
        name = user_input.split()[-1]  # 假设用户输入 "My name is X"
        user_identity['name'] = name
        return f"Hello, {name}! How can I assist you today?"
    elif intent == "transaction":
        return "Please provide more details for your order."
    elif intent == "question":
        return "Let me check the information for you."
    elif intent == "small_talk":
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
        
        intent = identify_intent(user_input)
        response = handle_intent(intent, user_input)
        print("Bot:", response)

# 启动聊天
chat()
