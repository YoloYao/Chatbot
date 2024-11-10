# Chatbot
Generative conversation robot based on NLP

/chatbot_project/                # 项目根目录<br>
├── /data/                       # 存放数据集的文件夹<br>
│   ├── /train/                  # 拆分后的训练数据集<br>
│   ├── /test/                   # 拆分后的测试数据集<br>
│   ├── /validation/             # 拆分后的验证数据集<br>
│   ├── dataset.csv              # 已标记意图类型的原始数据集<br>
│   └── preprocessed_dataset.csv # 预处理后的数据<br>
│   
├── /models/                     # 存放训练后的模型<br>
│   ├── language_model.pkl       # 存放训练模型<br>
│   └── vector_space.pkl         # 保存包含模型权重的向量空间<br>
├── /scripts/                    # 存放脚本文件<br>
│   ├── preprocess.py            # 数据预处理模块<br>
│   ├── train.py                 # 模型训练模块<br>
│   └── utils.py                 # 工具函数<br>
├── /config/                     # 配置文件<br>
│   ├── constrants.py            # 存放代码相关路径及常量<br>
│   └── intent_labels.json       # 意图种类配置文件<br>
├── /logs/                       # 训练日志<br>
│   └── training_log.txt         # 训练过程中的日志<br>
├── main.py                      # 程序主入口，用于启动对话机器人<br>
├── requirements.txt             # 依赖包列表<br>
└── README.md                    # 项目介绍及使用说明<br>