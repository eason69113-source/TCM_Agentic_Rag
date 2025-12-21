import os

from dotenv import load_dotenv

load_dotenv(override=True)

DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_API_URL=os.getenv("DASHSCOPE_API_URL")

ZHIPUAI_API_KEY=os.getenv("ZHIPUAI_API_KEY")
ZHIPUAI_API_URL=os.getenv("ZHIPUAI_API_URL")

