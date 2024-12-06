from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
import re
import sys
import os
# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from scripts.chat import ChatController
from scripts.utils import Utils
from config.constants import Constants
app = Flask(__name__)
CORS(app)  # 启用跨域支持


chat_controller = ChatController()
# 读取问题模型及向量空间
vectorizer = Utils.read_serialize_data(
            Constants.MODELS_FILE_DIR + Constants.QUESTION_LABEL + Constants.VECTOR_FILE_NAME)
model = Utils.read_serialize_data(
            Constants.MODELS_FILE_DIR + Constants.QUESTION_LABEL + Constants.MODELS_FILE_NAME)
# 获取问答模型集合
trans_model_file_name = Constants.MODELS_FILE_DIR + \
            Constants.QUESTION_LABEL + Constants.TRANS_MODEL_FILE_PATH
question_models = Utils.read_serialize_data(trans_model_file_name)

@app.route('/validateUsername', methods=['POST'])
def validateUsername():
    data = request.get_json()
    username = data.get('username', '')
    if chat_controller.service_authenticate(username):
        return jsonify({"success": True, "message": "Validate success"})
    else:
        return jsonify({"success": False, "message": "Invalid user name"}), 400


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')
    username = data.get('username')
    response = chat_controller.service_chat(vectorizer, model, question_models, message, username)
    if response == "":
        response = "Sorry. I couldn't find a suitable response. Can you rephrase your question?"
    return jsonify({"reply": response})



if __name__ == '__main__':
    app.run(debug=True)
