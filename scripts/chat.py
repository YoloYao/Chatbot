from scripts.utils import Utils
from scripts.preprocess import PreProcessController
from models.model import Model
from config.constants import Constants
from config.contexts import Contexts
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class ChatController:
    def __init__(self):
        self.context = {}

    def add_to_context(self, key, value):
        self.context[key] = value

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
                print("_" * Constants.SPLIT_LINE_LENGTH)
                print(f"Bot: Hello, {filtered_input}!")
                return filtered_input
            else:
                print("_" * Constants.SPLIT_LINE_LENGTH)
                print("No identification, please re-enter:")
                check_counter += 1
        # 超过3次错误输入，结束程序
        print("_" * Constants.SPLIT_LINE_LENGTH)
        print("You have made too many incorrect attempts. Goodbye.")
        return "exit"

    # 根据买票场景进行上下文处理
    def answer_by_context(self, user_input):
        # 检查上下文中是否已经存储了一些信息
        if any(key in user_input for key in Contexts.BOOK_KEYS):
            response = Contexts.DESTINATION_HINT
        elif "to" in user_input and "destination" not in self.context:
            destination = user_input.split("to")[-1].strip()
            self.add_to_context("destination", destination)
            response = Contexts.TIME_HINT.format(destination.capitalize())
        elif "at" in user_input and "depart_time" not in self.context and "destination" in self.context:
            depart_time = user_input.split("at")[-1].strip()
            self.add_to_context("depart_time", depart_time)
            response = Contexts.NUM_HINT.format(depart_time)
        elif user_input.isdigit() and "tickets_num" not in self.context and "depart_time" in self.context and "destination" in self.context:
            tickets_num = int(user_input)
            self.add_to_context("tickets_num", tickets_num)
            response = Contexts.CONFIRM_HINT.format(
                tickets_num, self.context['destination'].capitalize(), self.context['depart_time'])
        elif "yes" in user_input and "tickets_num" in self.context and "depart_time" in self.context and "destination" in self.context:
            response = Contexts.SUCCESS_HINT
            # 清空上下文，模拟结束会话
            self.context.clear()
        elif "no" in user_input.lower() and "tickets_num" in self.context and "depart_time" in self.context and "destination" in self.context:
            response = Contexts.STOP_HINT
            # 重新开始对话
            self.context.clear()
        else:
            response = ""

        return response

    # 根据意图回答问题
    def answer_question(self, user_input, model, intent_num):
        response = self.answer_by_context(user_input)
        if response != "":
            return response
        # 转换输入内容
        input_vector = model.vectorizer.transform([user_input])
        query_tf = model.tf_transformer.transform(input_vector)
        # print(f"Query: {user_input}")
        # 计算与每篇文档的余弦相似度
        cosine_similarities = cosine_similarity(
            query_tf, model.tf_data).flatten()
        # 对结果进行排序
        ranked_doc_indices = cosine_similarities.argsort()[::-1]
        # 找出对于答案集中的答案
        intent_menu = Utils.read_json(Constants.INTENT_LABEL_FILEPATH)
        label = Constants.ANSWER_LABEL
        # 找到问题文件中的对应编号
        question_file_path = Constants.MODELS_FILE_DIR + Constants.QUESTION_LABEL + '/' + \
            Constants.QUESTION_LABEL + '_' + \
            intent_menu[str(intent_num)]+Constants.DATA_CSV_FILE_NAME
        q_data = Utils.read_csv(question_file_path)
        num_data = q_data[Constants.NUMBER_LABEL]
        num_list = []
        similar = []
        # print("Top relevant questions:")
        for index in ranked_doc_indices[:3]:
            num_list.append(num_data[index])
            similar.append(cosine_similarities[index])
            # print(f"Question {
            # index + 1} (Score: {cosine_similarities[index]:.4f}): {model.data[index]}")
        # 到答案文件中找对应编号的答案
        answer_file_path = Constants.MODELS_FILE_DIR + label + '/' + label + \
            '_' + intent_menu[str(intent_num)]+Constants.DATA_CSV_FILE_NAME
        data = Utils.read_csv(answer_file_path)
        df = pd.DataFrame(data)
        filtered_data = []
        for num in num_list:
            filtered_answer = df[df[Constants.NUMBER_LABEL] ==
                                 num][Constants.ANSWER_LABEL].reset_index(drop=True)
            filtered_data.append(filtered_answer.iloc[0])
        # 打印排名前 5 的文档

        # print("Top relevant answers:")
        # for index in range(len(num_list)):
            # print(f"Answer {
            # num_list[index]} (Score: {similar[index]:.4f}): {filtered_data[index]}")
        # //important print
        # print(f"[Score: {cosine_similarities[ranked_doc_indices[0]]:.4f}]")
        # 相似度为0时返回默认回答
        if cosine_similarities[ranked_doc_indices[0]] == 0:
            return Constants.DEFAULT_ANSWER
        return str(filtered_data[0])

    # 使用模型对输入内容进行意图预测
    def predict_intent(self, user_input, vectorizer, model):
        input_vector = vectorizer.transform([user_input])
        tf_transformer = TfidfTransformer()
        tf_input = tf_transformer.fit_transform(input_vector)
        intent_num = model.predict(tf_input)[0]
        intent_menu = Utils.read_json(Constants.INTENT_LABEL_FILEPATH)
        # 显示意图
        # //important print
        # print(f"[Intention:{intent_menu[str(intent_num)]}]")
        return intent_num

    def chat(self):
        preprocessor = PreProcessController()
        preprocessor.init()
        print("Welcome to the chatbot! Type 'exit' to quit.")
        user_name = self.authenticate()
        # user_name = "Dylan"
        # 鉴权错误次数过多或选择退出，则结束程序
        if user_name == "exit" or user_name == "":
            return
        # 读取问题模型及向量空间
        vectorizer = Utils.read_serialize_data(
            Constants.MODELS_FILE_DIR + Constants.QUESTION_LABEL + Constants.VECTOR_FILE_NAME)
        model = Utils.read_serialize_data(
            Constants.MODELS_FILE_DIR + Constants.QUESTION_LABEL + Constants.MODELS_FILE_NAME)

        # 获取问答模型集合
        trans_model_file_name = Constants.MODELS_FILE_DIR + \
            Constants.QUESTION_LABEL + Constants.TRANS_MODEL_FILE_PATH
        question_models = Utils.read_serialize_data(trans_model_file_name)

        while True:
            user_input = input(f"{user_name}: ")
            filtered_input = Utils.clean_input(user_input)
            if filtered_input == None:
                continue
            if filtered_input.lower() == "exit":
                print("Goodbye!")
                break
            # 预处理输入内容
            # important print
            # print(f"[Berore:{filtered_input}]")
            filtered_input = " ".join(
                preprocessor.preprocess_data(filtered_input))
            # print(f"[After:{filtered_input}]")
            # 意图分析
            intent_num = self.predict_intent(filtered_input, vectorizer, model)
            # 生成回答内容
            response = self.answer_question(
                filtered_input, question_models[intent_num], intent_num)
            # 回答问题语句中需要动态加入用户名
            print("Bot:", response.format(user_name))
            print("_" * Constants.SPLIT_LINE_LENGTH)
