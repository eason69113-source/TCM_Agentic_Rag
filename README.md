![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.125.0-009688)
![Gradio](https://img.shields.io/badge/Gradio-3.50.2-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Stars](https://img.shields.io/github/stars/你的ID/仓库名?style=social)
# 《黄帝外经》智能问答系统
- **这是一个入门级的agentic_rag项目，非常适合新手来了解rag和agent的原理及应用，并且实现了端到端的应用，帮助新手发布第一个属于自己的rag网页，对于刚入门的新手很友好**
- 一个基于RAG（Retrieval-Augmented Generation）技术的中医古籍智能问答系统，支持《黄帝外经》的原文检索、白话翻译和专家解读。
<p align="center">
  <img src="png/1.png" width="100%"/>
</p>

## 🌟 项目特色

- **多模态检索**：将原文、直译、解要三种模式的检索结合输出
- **用户系统**：完整的用户注册、登录、会话管理功能
- **流式响应**：支持实时流式输出，提升用户体验
- **双界面支持**：提供FastAPI接口和Gradio可视化界面
- **并发处理**：支持多工具并行调用，提高响应效率
- **记忆功能**：支持用户偏好和对话历史的持久化存储

## 🏗️ 系统架构

```mermaid
graph TD
    A[用户输入] --> B[Agent处理]
    B --> C{工具选择}
    C -->|文档检索| D[向量数据库]
    C -->|网络搜索| E[互联网]
    D --> F[文档评分]
    F -->|相关| G[生成回答]
    F -->|不相关| H[重写查询]
    H --> B
    G --> I[输出结果]
```

## 📋 功能模块

### 核心组件
- **ancient_rag.py**: RAG流程核心，包含状态图定义和节点逻辑
- **main.py**: FastAPI服务，提供RESTful API接口
- **webUI.py**: Gradio可视化界面，支持Web交互
- **tools.py**: 工具配置，包含检索器和搜索工具

### 数据处理
- **pdf2json.py**: PDF文档OCR识别和结构化处理
- **save_db.py**: 构建FAISS向量数据库
- **config.py**: 系统配置管理

## 🚀 快速开始（手动配置）

### 环境准备

```bash
# 克隆项目
git clone https://github.com/eason69113-source/TCM_Agentic_Rag.git
cd TCM_Agentic_Rag

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

创建 `.env` 文件并配置以下参数：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_API_URL=https://dashscope.aliyuncs.com/api/v1
ZHIPUAI_API_KEY=your_zhipuai_api_key
ZHIPUAI_API_URL=https://open.bigmodel.cn/api/paas/v4
DB_URI=postgresql://username:password@localhost:5432/database_name
```

### 数据准备（已经处理好在faiss_db）

1. **处理PDF文档**（可选）：
```bash
python pdf2json.py
```

2. **构建向量数据库**（可选）：
```bash
python save_db.py
```

### 数据库安装

1. **下载PostGreSQL15.15**
2. **创建库 & 用户**
3. **根据vector.v0.8.0-pg15.14文件中的readme进行包的补充**

### 启动服务

**步骤一：API服务**
```bash
python main.py
# 访问 http://localhost:8000/docs 查看API文档
```

**步骤二：Web界面**
```bash
python webUI.py
# 访问 http://localhost:7860 使用Web界面
```

## 🚀 快速开始（docker管理）

### 环境准备

```bash
# 克隆项目
git clone https://github.com/eason69113-source/TCM_Agentic_Rag.git
cd TCM_Agentic_Rag
```

### 配置环境变量

创建或进入 `.env` 文件并配置以下参数：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_API_URL=https://dashscope.aliyuncs.com/api/v1
ZHIPUAI_API_KEY=your_zhipuai_api_key
ZHIPUAI_API_URL=https://open.bigmodel.cn/api/paas/v4
DB_URI=postgresql://username:password@localhost:5432/database_name
```

进入`docker-compose.yml`文件并配置以下参数：

```environment
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_API_URL=https://dashscope.aliyuncs.com/api/v1
ZHIPUAI_API_KEY=your_zhipuai_api_key
ZHIPUAI_API_URL=https://open.bigmodel.cn/api/paas/v4
DB_URI=postgresql://username:password@localhost:5432/database_name
```

### 启动系统

```bash
docker compose up --build -d
```

### 创建用户表（只做一次

```bash
docker exec -it tcm_rag_db psql -U postgres -d tcm_db -c "CREATE TABLE IF NOT EXISTS users (id UUID PRIMARY KEY, username VARCHAR(255) UNIQUE NOT NULL, password_hash TEXT NOT NULL);"
```

### 使用系统

- 打开浏览器：http://localhost:7860
- 注册账号 → 登录 → 开始提问《黄帝外经》相关问题
- 右上角“退出登录”可退出

## 📖 使用说明

### API调用示例

```python
import requests

# 用户登录
response = requests.post("http://localhost:8000/auth/login", 
    json={"username": "user", "password": "pass"})
user_id = response.json()["user_id"]

# 发送消息
response = requests.post("http://localhost:8000/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "什么是阴阳五行？"}],
        "userId": user_id,
        "stream": False
    })
print(response.json()["choices"][0]["message"]["content"])
```

### Web界面功能

1. **用户系统**：注册、登录、会话管理
2. **多轮对话**：支持上下文连续对话
3. **历史记录**：查看和管理历史会话
4. **实时响应**：流式输出，无需等待

## 🔧 技术栈

- **后端框架**: FastAPI + Uvicorn
- **AI模型**: 通义千问 + 智谱AI
- **向量数据库**: FAISS + PostgreSQL
- **前端界面**: Gradio
- **文档处理**: PaddleOCR + LangChain
- **并发处理**: asyncio + ThreadPoolExecutor

## 📁 项目结构

```
TCM_Agentic_Rag/
├── ancient_rag.py         # RAG核心流程
├── main.py                # FastAPI服务
├── webUI.py               # Gradio界面
├── config.py              # 配置管理
├── requirements.txt       # 项目依赖
├── .env                   # 环境变量
├── vector.v0.8.0-pg15.14  # pg15的vector补充包
├── data/                  # 数据
│   ├── hdwj.json/         # json数据
│   ├── test.py/           # 测试向量数据库
├── faiss_db/              # 向量数据库
│   ├── raw/               # 原文索引
│   ├── trans/             # 直译索引
│   └── note/              # 解要索引
├── utils/                 # 工具
│   ├── tools.py /         # 智能体工具配置
│   ├── pdf2json.py /      # PDF处理
│   └── save_db.py/        # 向量数据库构建
│   └── log.py/            # 日志输出构建
├── prompts/               # 提示词模板
│   ├── prompt_template_agent.txt
│   ├── prompt_template_grade.txt
│   ├── prompt_template_rewrite.txt
│   └── prompt_template_generate.txt
└── output/                # 日志输出
    └── app.log
```

## 🔍 核心特性详解

### 智能检索
- **混合检索**：结合稀疏检索和密集检索
- **权重配置**：原文(0.2) + 直译(0.5) + 解要(0.3)
- **相关性评分**：自动评估文档相关性

### 流程控制
- **自动重写**：查询不相关时自动重写
- **重试机制**：最多重写3次，避免无限循环
- **错误处理**：完善的异常捕获和恢复机制

### 性能优化
- **连接池管理**：PostgreSQL连接池监控
- **并发处理**：多工具并行调用
- **缓存机制**：提示词模板缓存

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

1. Fork 本项目
2. 创建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢阿里云通义千问提供的大模型支持
- 感谢智谱AI提供的AI服务
- 感谢PaddleOCR提供的文字识别能力


⭐ 如果这个项目对你有帮助，请给个Star支持一下！
