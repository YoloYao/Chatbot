# Chatbot
Generative conversation robot based on NLP

/chatbot_project/                # 项目根目录
├── /data/                       # 存放数据集的文件夹
│   ├── /train/                  # 拆分后的训练数据集
│   ├── /test/                   # 拆分后的测试数据集
│   ├── /validation/             # 拆分后的验证数据集
│   ├── dataset.csv              # 已标记意图类型的原始数据集
│   └── preprocessed_dataset.csv # 预处理后的数据
│   
├── /models/                     # 存放训练后的模型
│   ├── language_model.pkl       # 存放训练模型
│   └── vector_space.pkl         # 保存包含模型权重的向量空间
├── /scripts/                    # 存放脚本文件
│   ├── preprocess.py            # 数据预处理模块
│   ├── train.py                 # 模型训练模块
│   └── utils.py                 # 工具函数
├── /config/                     # 配置文件
│   ├── constrants.py            # 存放代码相关路径及常量
│   └── intent_labels.json       # 意图种类配置文件
├── /logs/                       # 训练日志
│   └── training_log.txt         # 训练过程中的日志
├── main.py                      # 程序主入口，用于启动对话机器人
├── requirements.txt             # 依赖包列表
└── README.md                    # 项目介绍及使用说明