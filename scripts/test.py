import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from spellchecker import SpellChecker
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split

# 下载所需的NLTK资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 假设的数据预处理方法

# 1. 清洗文本
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # 去除HTML标签
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 去除特殊字符和数字
    text = text.lower().strip()  # 转换为小写
    return text

# 2. 分词
def tokenize(text):
    return word_tokenize(text)

# 3. 去除停用词
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# 4. 词形还原
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# 5. 拼写纠正
def correct_spelling(tokens):
    spell = SpellChecker()
    corrected_tokens = [spell.correction(token) for token in tokens]
    return corrected_tokens

# 6. 生成n-gram特征
def generate_ngrams(tokens, n=2):
    n_grams = list(ngrams(tokens, n))
    return [' '.join(grams) for grams in n_grams]

# 7. 移除少见词和高频词
def remove_rare_and_frequent_words(tokens, min_freq=2, max_freq_ratio=0.8):
    word_counts = Counter(tokens)
    total_words = len(tokens)
    return [word for word in tokens if min_freq <= word_counts[word] <= total_words * max_freq_ratio]

# 8. 保存预处理后的数据
def save_preprocessed_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# 9. 数据集拆分
def split_dataset(X, y, test_size=0.2, validation_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size)
    return X_train, X_val, X_test, y_train, y_val, y_test

# 主方法：将所有步骤串联起来
def main_pipeline(text_data):
    preprocessed_data = []
    
    for text in text_data:
        # 1. 清洗文本
        cleaned_text = clean_text(text)
        
        # 2. 分词
        tokens = tokenize(cleaned_text)
        
        # 3. 去除停用词
        tokens = remove_stopwords(tokens)
        
        # 4. 词形还原
        tokens = lemmatize(tokens)
        
        # 5. 拼写纠正
        tokens = correct_spelling(tokens)
        
        # 6. 生成 n-gram 特征（可选，使用二元组为例）
        ngrams_tokens = generate_ngrams(tokens, n=2)
        
        # 7. 移除少见词和高频词
        final_tokens = remove_rare_and_frequent_words(tokens)
        
        # 将预处理后的数据存储
        preprocessed_data.append(final_tokens)
    
    # 假设 y 是标签（如果是分类任务），这里只是示例
    y = [0] * len(preprocessed_data)  # 占位符标签，实际中应使用真实标签

    # 8. 数据集拆分
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(preprocessed_data, y)
    
    # 9. 保存预处理后的训练数据
    save_preprocessed_data((X_train, y_train), 'train_data.pkl')
    save_preprocessed_data((X_val, y_val), 'val_data.pkl')
    save_preprocessed_data((X_test, y_test), 'test_data.pkl')

    print("数据预处理完成并保存至本地文件。")

# 示例文本数据
text_data = [
    "This is the first sentence. It's quite simple!",
    "Here's the second sentence, which is a bit longer.",
    "The third one is shorter."
]

# 运行主预处理管道
main_pipeline(text_data)