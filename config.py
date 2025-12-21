import os
from zai import ZhipuAiClient
from langchain_openai import ChatOpenAI
from env_utils import *
from langchain_community.embeddings import DashScopeEmbeddings

class Config:
    """统一的配置类，集中管理所有常量"""
    # prompt文件路径
    PROMPT_TEMPLATE_TXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE = "prompts/prompt_template_generate.txt"

    llm1 = ChatOpenAI(model='qwen-max',
                  temperature=0.5,
                  extra_body={"enable_search": True},
                  api_key=DASHSCOPE_API_KEY,
                  base_url=DASHSCOPE_API_URL
)

    llm2 = ZhipuAiClient(api_key=ZHIPUAI_API_KEY)

    embed1 = DashScopeEmbeddings(model='text-embedding-v3', 
                                dashscope_api_key=DASHSCOPE_API_KEY)

    DB_URI = os.getenv("DB_URI", "postgresql://postgres:密码@localhost:5432/数据库名")

    HOST = "0.0.0.0"

    PORT = 8000
