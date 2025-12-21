from contextlib import asynccontextmanager
import re
import time
from typing import List, Optional, Tuple
import uuid
import json
import uvicorn
from fastapi.responses import JSONResponse, StreamingResponse
from utils.log import Logger
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from config import Config
from utils.tools import ToolConfig
from ancient_rag import (
    create_graph,
    save_graph_visualization,
    ConnectionPoolError,
    monitor_connection,
    ConnectionPool
)
import sys
from passlib.context import CryptContext

logger = Logger()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

conn_pool: ConnectionPool | None = None

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hash: str) -> bool:
    return pwd_context.verify(password, hash)

class RegisterRequest(BaseModel):
    username: str
    password: str

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionChoice]
    system_fingerprint: Optional[str] = None

def format_response(response):
    paragraphs = re.split(r'\n{2,}', response)
    formatted_paragraphs = []
    # 遍历每个段落进行处理
    for para in paragraphs:
        # 检查段落中是否包含代码块标记
        if '```' in para:
            # 将段落按照```分割成多个部分，代码块和普通文本交替出现
            parts = para.split('```')
            for i, part in enumerate(parts):
                # 检查当前部分的索引是否为奇数，奇数部分代表代码块
                if i % 2 == 1:  # 这是代码块
                    # 将代码块部分用换行符和```包围，并去除多余的空白字符
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            # 将分割后的部分重新组合成一个字符串
            para = ''.join(parts)
        else:
            # 否则，将句子中的句点后面的空格替换为换行符，以便句子之间有明确的分隔
            para = para.replace('. ', '.\n')

        formatted_paragraphs.append(para.strip())
    return '\n\n'.join(formatted_paragraphs)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, tool_config, conn_pool

    try:
        llm, embed = Config.llm1, Config.embed1
        tool_config = ToolConfig(embed=embed, llm=Config.llm2)

        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "connect_timeout": 5
        }

        conn_pool = ConnectionPool(
            conninfo=Config.DB_URI,
            max_size=20,
            min_size=2,
            kwargs=connection_kwargs,
            timeout=10
        )
        try:
            conn_pool.open()
            logger.info("数据库连接池已打开")
            logger.debug("数据库连接池已打开")
        except Exception as e:
            logger.error(f"数据库连接池打开失败: {e}")
            raise ConnectionPoolError("数据库连接池打开失败")
        
        monitor_thread = monitor_connection(conn_pool, interval=60)

        try:
            graph = create_graph(conn_pool, llm, embed, tool_config)
        except Exception as e:
            logger.error(f"创建图失败: {e}")
            print("错误: 创建图失败")
            sys.exit(1)

        save_graph_visualization(graph)

    except ConnectionPoolError as e:
        # 捕获连接池相关的异常
        logger.error(f"连接池错误: {e}")
        print(f"错误: 数据库连接池问题 - {e}")
        sys.exit(1)
    except RuntimeError as e:
        # 捕获其他运行时错误
        logger.error(f"初始化失败: {e}")
        print(f"错误: 初始化失败 - {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        # 捕获键盘中断
        print("\n被用户打断。再见！")
    except Exception as e:
        # 捕获未预期的其他异常
        logger.error(f"未知问题: {e}")
        print(f"错误: 发生未知错误 - {e}")
        sys.exit(1)

    yield

    if conn_pool and not conn_pool.closed:
            conn_pool.close()
            logger.info("数据库连接池已关闭")

    logger.info("服务器已关闭")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://120.55.90.221:7860", "http://120.55.90.221:8000"],  # 允许所有来源（*）在调试阶段是最简单的，生产环境应限制为前端地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法 (GET, POST, OPTIONS 等)
    allow_headers=["*"],  # 允许所有请求头
)

async def handle_non_stream_response(user_input, graph, tool_config, config):

    content = None
    try:
        events = graph.stream({"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}, config)
        for event in events:
            for value in event.values():
                if "messages" not in value or not isinstance(value["messages"], list):
                    logger.warning("回答中没有有效的消息")
                    continue

                last_message = value["messages"][-1]

                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        if isinstance(tool_call, dict) and "name" in tool_call:
                            logger.info(f"调用工具: {tool_call['name']}")

                    continue

                if hasattr(last_message, "content"):
                    content = last_message.content

                    if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
                        tool_name = last_message.name
                        logger.info(f"工具输出[{tool_name}]: {content}")

                    else:
                        logger.info(f"最终输出：{content}")

    except Exception as e:
        logger.error(f"处理响应时发生错误: {e}")
        print("处理响应时发生错误")

    formatted_response = str(format_response(content)) if content else "没有响应"

    logger.info(f"格式化输出结果：{formatted_response}")

    try:
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=formatted_response
                    ),
                    finish_reason="stop"
                )
            ]
        )

    except Exception as e:
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="处理响应时发生错误"
                    ),
                    finish_reason="error"
                )
            ]
        )

    logger.info(f"响应结果：\n{response}")
    return JSONResponse(content=response.model_dump())
          
async def handle_stream_response(user_input, graph, config):
    """
    处理流式响应的异步函数，生成并返回流式数据。

    Args:
        user_input (str): 用户输入的内容。
        graph: 图对象，用于处理消息流。
        config (dict): 配置参数，包含线程和用户标识。

    Returns:
        StreamingResponse: 流式响应对象，媒体类型为 text/event-stream。
    """
    async def generate_stream():
        """
        内部异步生成器函数，用于产生流式响应数据。

        Yields:
            str: 流式数据块，格式为 SSE (Server-Sent Events)。

        Raises:
            Exception: 流生成过程中可能抛出的异常。
        """
        try:
            # 生成唯一的 chunk ID
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            # 调用 graph.stream 获取消息流
            stream_data = graph.stream(
                {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0},
                config,
                stream_mode="messages"
            )
            # 遍历消息流中的每个数据块
            for message_chunk, metadata in stream_data:
                try:
                    # 获取当前节点名称
                    node_name = metadata.get("langgraph_node") if metadata else None
                    # 仅处理 generate 和 agent 节点
                    if node_name in ["generate", "agent"]:
                        # 获取消息内容，默认空字符串
                        chunk = getattr(message_chunk, 'content', '')
                        # 记录流式数据块日志
                        logger.info(f"Streaming chunk from {node_name}: {chunk}")
                        # 产出流式数据块
                        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                except Exception as chunk_error:
                    # 记录单个数据块处理异常
                    logger.error(f"Error processing stream chunk: {chunk_error}")
                    continue

            # 产出流结束标记
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        except Exception as stream_error:
            # 记录流生成过程中的异常
            logger.error(f"Stream generation error: {stream_error}")
            # 产出错误提示
            yield f"data: {json.dumps({'error': 'Stream processing failed'})}\n\n"

    # 返回流式响应对象
    return StreamingResponse(generate_stream(), media_type="text/event-stream")
        

# 依赖注入函数，用于获取 graph 和 tool_config
async def get_dependencies() -> Tuple[any, any]:
    """
    依赖注入函数，用于获取 graph 和 tool_config。

    Returns:
        Tuple: 包含 (graph, tool_config) 的元组。

    Raises:
        HTTPException: 如果 graph 或 tool_config 未初始化，则抛出 500 错误。
    """
    if not graph or not tool_config:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return graph, tool_config


@app.post("/auth/register")
def register_user(req: RegisterRequest):
    try:
        with conn_pool.connection() as conn:
            cur = conn.cursor()

            cur.execute(
                "SELECT 1 FROM users WHERE username=%s",
                (req.username,)
            )
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="用户名已存在")

            user_id = uuid.uuid4()
            pwd_hash = hash_password(req.password)

            cur.execute(
                "INSERT INTO users (id, username, password_hash) VALUES (%s, %s, %s)",
                (user_id, req.username, pwd_hash)
            )

        return {"msg": "注册成功"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注册异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="注册失败")


class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/auth/login")
def login_user(req: LoginRequest):
    try:
        with conn_pool.connection() as conn:
            cur = conn.cursor()

            cur.execute(
                "SELECT id, password_hash FROM users WHERE username=%s",
                (req.username,)
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=401, detail="用户名或密码错误")

            user_id, pwd_hash = row
            if not verify_password(req.password, pwd_hash):
                raise HTTPException(status_code=401, detail="用户名或密码错误")

        return {
            "user_id": str(user_id),
            "username": req.username
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"登录异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="登录失败")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, dependencies: Tuple[any, any] = Depends(get_dependencies)):
    try:
        graph, tool_config = dependencies
        if not request.messages or not request.messages[-1].content:
            logger.info("请求消息为空，请检查输入")
            raise HTTPException(status_code=400, detail="Messages cannot be empty or invalid")
        user_input = request.messages[-1].content
        logger.info(f"用户输入：{user_input}")

        config = {
            "configurable":{
                "thread_id": f"{getattr(request, 'userId', 'unknown')}@@{getattr(request, 'conversationId', 'default')}",
                "user_id": getattr(request, 'userId', 'unknown')
            }
        }

        if request.stream:
            return await handle_stream_response(user_input, graph, config)
        return await handle_non_stream_response(user_input, graph, tool_config, config)

    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    logger.info(f"Start the server on port {Config.PORT}")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
