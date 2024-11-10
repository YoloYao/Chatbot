from scripts.utils import Utils
from scripts.preprocess import PreProcessController
from config.constants import Constants
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class ChatController:
    # 身份鉴权
    def authenticate(self):
        name_data = Utils.read_json(Constants.USER_LIST_FILEPATH)
        whitelist = name_data.get(Constants.WHITELIST_LABEL, [])
        check_counter = 0
        while check_counter < 3:
            user_input = input("Please enter your name: ")
            filtered_input = Utils.clean_input(user_input)
            if filtered_input == None:
                continue
            if filtered_input.lower() == "exit":
                print("Goodbye!")
                return "exit"
            if filtered_input in whitelist:
                print(f"Hello, {filtered_input}!")
                return filtered_input
            else:
                print("No identification, please re-enter:")
                check_counter += 1
        # 超过3次错误输入，结束程序
        print("You have made too many incorrect attempts. Goodbye.")
        return "exit"

    # 根据意图回答问题
    def answer_question(self, intent, user_input, answer_vectorizer, tfidf_transformer, answer_tf_data, filtered_answers):
        # 转换输入内容
        input_vector = answer_vectorizer.transform([user_input])
        query_tf = tfidf_transformer.transform(input_vector)
        # 计算与每篇文档的余弦相似度
        cosine_similarities = cosine_similarity(
            query_tf, answer_tf_data).flatten()

        # 对结果进行排序
        ranked_doc_indices = cosine_similarities.argsort()[::-1]
        # 打印排名前 5 的文档
        # print(f"Query: {user_input}")
        # print("\nTop relevant documents:")

        # for index in ranked_doc_indices[:5]:
        #     print(f"Document {
        #         index + 1} (Score: {cosine_similarities[index]:.4f}): {filtered_answers[index]}")
        print(f"[Score: {cosine_similarities[ranked_doc_indices[0]]:.4f}]")
        # 相似度为0时返回默认回答
        if cosine_similarities[ranked_doc_indices[0]] == 0:
            return Constants.DEFAULT_ANSWER
        return filtered_answers[ranked_doc_indices[0]]

    # 使用模型对输入内容进行意图预测
    def predict_intent(self, user_input, vectorizer, model):
        input_vector = vectorizer.transform([user_input])
        tf_transformer = TfidfTransformer()
        tf_input = tf_transformer.fit_transform(input_vector)
        intent_num = model.predict(tf_input)[0]
        intent_menu = Utils.read_json(Constants.INTENT_LABEL_FILEPATH)
        # 显示意图
        print(f"[{intent_menu[str(intent_num)]}]")
        return intent_num

    def chat(self):
        preprocessor = PreProcessController()
        preprocessor.init()
        print("Welcome to the chatbot! Type 'exit' to quit.")
        # user_name = self.authenticate()
        user_name = "Dylan"
        # 鉴权错误次数过多或选择退出，则结束程序
        if user_name == "exit" or user_name == "":
            return
        # 读取模型及向量空间
        vectorizer = Utils.read_serialize_data(
            Constants.MODELS_FILE_DIR + Constants.VECTOR_FILE_NAME)
        model = Utils.read_serialize_data(
            Constants.MODELS_FILE_DIR + Constants.MODELS_FILE_NAME)
        # 读取答案模型###########################
        file_path = './data/answer.csv'
        data = Utils.read_csv(file_path)
        # answers = data[Constants.ANSWER_LABEL]
        df = pd.DataFrame(data)
        # 过滤出 intent 等于 intent 的所有 answer
        filtered_answers = data[Constants.ANSWER_LABEL]
        # filtered_answers = df[df["intent"] == intent]["answer"]
        processed_answers = preprocessor.preprocess_list_data(filtered_answers)
        # 构建文档-词项矩阵（Term-Document Matrix）
        answer_vectorizer = CountVectorizer()  # 或使用其他加权方式，如 TfidfVectorizer
        answer_vec_data = answer_vectorizer.fit_transform(processed_answers)
        # 使用 TF-IDF 转换词频矩阵
        tfidf_transformer = TfidfTransformer()
        answer_tf_data = tfidf_transformer.fit_transform(answer_vec_data)
        ######################################
        while True:
            user_input = input(f"{user_name}: ")
            filtered_input = Utils.clean_input(user_input)
            if filtered_input == None:
                continue
            if filtered_input.lower() == "exit":
                print("Goodbye!")
                break
            # 预处理输入内容
            # print(f"11111{filtered_input}")
            filtered_input = " ".join(
                preprocessor.preprocess_data(filtered_input))
            # print(f"22222{filtered_input}")
            intent_num = self.predict_intent(filtered_input, vectorizer, model)
            response = self.answer_question(intent_num, filtered_input,
                                            answer_vectorizer, tfidf_transformer, answer_tf_data, filtered_answers)
            print("Bot:", response)
