import sys
import threading
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Literal, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langgraph.store.base import BaseStore 
from langchain_core.runnables import RunnableConfig
from html import escape
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import tools_condition, ToolNode
import threading
import time
from psycopg import OperationalError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from psycopg_pool import ConnectionPool
from config import Config
from utils.tools import ToolConfig
from utils.log import Logger

logger = Logger()

class ConnectionPoolError(Exception):
    """自定义异常，表示数据库连接池初始化或状态异常"""
    pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(OperationalError))
def test_connection(conn_pool: ConnectionPool) -> bool:
    with conn_pool.getconn() as conn: 
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
            if result != (1,):
                raise ConnectionPoolError("数据库连接池状态异常")
    return True

def monitor_connection(conn_pool: ConnectionPool, interval: int = 60):

    def _monitor():
        while not conn_pool.closed:
            try:
                stats = conn_pool.get_stats()
                total = stats.get('pool_size', 0)
                available = stats.get('pool_available', 0)
                active = total - available
                max_size = conn_pool.max_size
                logger.info(f"数据库连接池状态: 总数: {total}, 可用: {available}, 活动: {active}")
                if active > max_size * 0.8:
                    logger.warning(f"数据库连接池活动连接数过高: {active} / {max_size}")
                    
            except Exception as e:
                logger.error(f"数据库连接池异常: {e}")

            time.sleep(interval)

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread


# 定义消息状态类，使用TypedDict进行类型注解
class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    relevance_score: Annotated[Optional[str], "Relevance score of retrieved documents, 'yes' or 'no'"]
    rewrite_count: Annotated[int, "Number of times query has been rewritten"]

# 定义文档相关度分数类，使用Pydantic进行类型注解
class DocumentRelevanceScore(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# 重定义ToolNode，支持并发处理工具调用
class ParallelToolNode(ToolNode):
    def __init__(self, tools, max_workers: int=5):
        super().__init__(tools)
        self.max_workers = max_workers

    def _run_single_tool(self, tool_call: dict, tool_map: dict) -> ToolMessage:
        try:
            tool_name = tool_call["name"]
            tool = tool_map.get(tool_name)
            if not tool:
                raise ValueError(f"Tool '{tool_name}' not found")

            result = tool.invoke(tool_call["args"])
            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name
            )
        
        except Exception as e:
            logger.error(f"调用工具时发生错误: {e}")
            return ToolMessage(
                content=f"Error: {e}",
                tool_call_id=tool_call["id"],
                name=tool_call.get("name", "unknown")
            )
        
    def __call__(self, state: dict) -> dict:
        logger.info(f"使用并行工具处理问题")
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        if not tool_calls:
            logger.warning("没有工具调用")
            return {"messages": []}
        
        tool_map = {tool.name: tool for tool in self.tools}

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tool = {
                executor.submit(self._run_single_tool, tool_call, tool_map): tool_call
                for tool_call in tool_calls
            }

            for future in as_completed(future_to_tool):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"处理工具调用时发生错误: {e}")
                    tool_call = future_to_tool[future]
                    results.append(ToolMessage(
                        content=f"Error: {e}",
                        tool_call_id=tool_call["id"],
                        name=tool_call.get("name", "unknown")
                    ))

        logger.info(f"处理工具调用完成,包括{len(results)}个工具")
        return {"messages": results}




def filter_messages(messages: list) -> list:
    filtered = [msg for msg in messages if msg.__class__.__name__ in ["HumanMessage", "AIMessage"]]
    return filtered[-5:] if len(filtered) > 5 else filtered

def store_memory(question: BaseMessage, config: RunnableConfig, store: BaseStore) -> str:
    namespace = ("memories", config["configurable"]["user_id"])
    try:
        memories = store.search(namespace, query=str(question.content))
        user_info = "\n".join([d.value["data"] for d in memories])

        if "记住" in question.content.lower():
            memory = escape(question.content)
            store.put(namespace, uuid.uuid4(), {"data": memory})
            logger.info(f"记忆已存储: {memory}")

        return user_info
    
    except Exception as e:
        logger.error(f"存储记忆时发生错误: {e}")
        return ""
    
def create_chain(llm, template_file: str, structured_output=None):
    if not hasattr(create_chain, "prompt_cache"):
        create_chain.prompt_cache = {}
        create_chain.lock = threading.Lock()

    try:
        
        if template_file in create_chain.prompt_cache:
            prompt = create_chain.prompt_cache[template_file]
            logger.info(f"使用缓存的提示: {template_file}")

        else:
            with create_chain.lock:
                if template_file not in create_chain.prompt_cache:
                    logger.info(f"创建新的缓存提示: {template_file}")
                    with open(template_file, "r", encoding="utf-8") as f:
                        template_text = f.read()

                    prompt = ChatPromptTemplate.from_template(template_text)

                    create_chain.prompt_cache[template_file] = prompt

                prompt = create_chain.prompt_cache[template_file]

        if structured_output:
            return prompt | llm.with_structured_output(structured_output)
    
        return prompt | llm  # 必须返回 prompt | llm 的组合
    
    except Exception as e:
        logger.error(f"创建链时发生错误: {e}")
        raise

def get_last_question(state: MessagesState) -> str:
    try:
        if not state.get("messages") or not isinstance(state["messages"], (list, tuple)) or len(state["messages"]) == 0:
            logger.warning("No valid messages found in state for getting latest question")
            return None
        
        for message in reversed(state["messages"]):
            if message.__class__.__name__ == "HumanMessage" and hasattr(message, "content"):
                return message.content
            
        logger.info("没有在状态中找到问题")
        return None
                
    except Exception as e:
        logger.error(f"获取问题时发生错误: {e}")
        return None

def agent(state: MessagesState, config: RunnableConfig, store: BaseStore, llm, tool_config: ToolConfig) -> dict:
    logger.info(f"使用agent模式处理问题")
    namespace = ("memories", config["configurable"]["user_id"])

    try:
        question = state["messages"][-1]
        question_text = question.content if hasattr(question, 'content') else str(question)
        user_info = store_memory(question, config, store)

        history_messages = filter_messages(state["messages"])
        messages = "\n".join([f"{'用户' if m.type=='human' else '助手'}: {m.content}" for m in history_messages])


        llm_with_tools = llm.bind_tools(tool_config.get_tools())
        agent_chain = create_chain(llm_with_tools, Config.PROMPT_TEMPLATE_TXT_AGENT)

        response = agent_chain.invoke({"question": question_text, "user_info": user_info, "messages": messages})

        return {"messages": [response]}
    
    except Exception as e:
        logger.error(f"处理问题时发生错误: {e}")
        return {"messages": [{"role": "system", "content": "处理请求时出错"}]}
    
def grade_documents(state: MessagesState, llm) -> dict:
    logger.info(f"使用文档评估模式处理问题")
    if not state.get("messages"):
        logger.error("消息状态是空的")
        return {"messages": [{"role": "system", "content": "没有找到有效的消息"}],
                "relevance_score": None}
    
    try:
        question = get_last_question(state)
        context = state["messages"][-1].content

        grade_chain = create_chain(llm, Config.PROMPT_TEMPLATE_TXT_GRADE, DocumentRelevanceScore)

        response = grade_chain.invoke({"question": question, "context": context})

        score = response.binary_score
        logger.info(f"文档评估结果: {score}")

        return {"messages": state["messages"], "relevance_score": score}
    
    except Exception as e:
        logger.error(f"处理文档评估时发生错误: {e}")
        return {"messages": [{"role": "system", "content": "评分过程中出错"}],
                "relevance_score": None}
    
def rewrite(state: MessagesState, llm) -> dict:
    logger.info(f"使用重写模式处理问题")
    try:
        question = get_last_question(state)
        rewrite_chain = create_chain(llm, Config.PROMPT_TEMPLATE_TXT_REWRITE)
        response = rewrite_chain.invoke({"question": question})

        rewrite_count = state.get("rewrite_count", 0) + 1
        return {"messages": [response], "rewrite_count": rewrite_count}
    
    except Exception as e:
        logger.error(f"处理重写时发生错误: {e}")
        return {"messages": [{"role": "system", "content": "重写过程中出错"}]}
    
def generate(state: MessagesState, llm) -> dict:
    logger.info(f"使用生成模式处理问题")
    try:
        question = get_last_question(state)
        context = state["messages"][-1].content

        generate_chain = create_chain(llm, Config.PROMPT_TEMPLATE_TXT_GENERATE)
        response = generate_chain.invoke({"question": question, "context": context})

        return {"messages": [response]}
    
    except Exception as e:
        logger.error(f"处理生成时发生错误: {e}")
        return {"messages": [{"role": "system", "content": "生成过程中出错"}]}

def route_after_tools(state: MessagesState, tool_config: ToolConfig) -> Literal["generate", "grade_documents"]:
    if not state.get("messages") or not isinstance(state["messages"], list):
        logger.error("消息状态是空的,自动跳转为生成模式")
        return "generate"
    
    try:
        last_message = state["messages"][-1]

        if not hasattr(last_message, "name") or last_message.name is None:
            logger.error("最后一条消息没有名称,自动跳转为生成模式")
            return "generate"
        
        tool_name = last_message.name
        if tool_name not in tool_config.get_tool_names():
            logger.error(f"未知工具 {tool_name} 不在工具列表中,自动跳转为生成模式")
            return "generate"

        target = tool_config.get_tool_routing_config().get(tool_name, "generate")
        logger.info(f"工具 {tool_name} 匹配到目标模式 {target}")
        return target
    
    except Exception as e:
        logger.error(f"处理工具匹配时发生错误: {e}, 自动跳转为生成模式")
        return "generate"
        
def route_after_grade(state: MessagesState) -> Literal["generate", "rewrite"]:
    if not isinstance(state, dict):
        logger.error("状态不是字典,自动跳转为重写模式")
        return "rewrite"
    
    if not state["messages"]:
        logger.warning("消息状态是空的,自动跳转为重写模式")
        return "rewrite"
    
    if "messages" not in state or not isinstance(state["messages"], (list,tuple)):
        logger.error("状态缺失消息字段,自动跳转为重写模式")
        return "rewrite"
    
    relevance_score = state.get("relevance_score")
    rewrite_count = state.get("rewrite_count", 0)
    logger.info(f"文档评估结果: {relevance_score}, 重写次数: {rewrite_count}")

    if rewrite_count >= 3:
        logger.info("重写次数达到上限,自动跳转为生成模式")
        return "generate"
    
    try:
        if not isinstance(relevance_score, str):
            logger.warning("文档评估结果不是字符串,自动跳转为重写模式")
            return "rewrite"
        
        if relevance_score.lower() == "yes":
            logger.info("文档评估结果为 'yes', 自动跳转为生成模式")
            return "generate"

        logger.info("文档评估结果为 'no'或其他值, 自动跳转为重写模式")
        return "rewrite"
    except Exception as e:
        logger.error(f"处理文档评估结果时发生错误: {e}, 自动跳转为重写模式")
        return "rewrite"
    
def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    """保存状态图的可视化表示。

    Args:
        graph: 状态图实例。
        filename: 保存文件路径。
    """
    # 尝试执行以下代码块
    try:
        # 以二进制写模式打开文件
        with open(filename, "wb") as f:
            # 将状态图转换为Mermaid格式的PNG并写入文件
            f.write(graph.get_graph().draw_mermaid_png())
        # 记录保存成功的日志
        logger.info(f"Graph visualization saved as {filename}")
    # 捕获IO错误
    except IOError as e:
        # 记录警告日志
        logger.warning(f"Failed to save graph visualization: {e}")

def create_graph(conn_pool: ConnectionPool, llm, embed, tool_config: ToolConfig) -> StateGraph:
    # 检查连接池是否为None或未打开
    if conn_pool is None or conn_pool.closed:
        logger.error("数据库连接池已关闭")
        raise ConnectionPoolError("数据库连接池已关闭")
    
    # 获取当前活动连接数和最大连接数
    try:
        stats = conn_pool.get_stats()
        total = stats.get('pool_size', 0)
        available = stats.get('pool_available', 0)
        active = total - available
        max_size = conn_pool.max_size
        if active >= max_size:
            logger.warning(f"数据库连接池活动连接数过高: {active} / {max_size}")
            raise ConnectionPoolError("连接池已耗尽，无可用连接")
        if not test_connection(conn_pool):
            raise ConnectionPoolError("数据库连接池测试失败")
        logger.info("数据池连接状态：ok， 测试成功")
    except Exception as e:
        logger.error(f"数据库连接池异常: {e}")
        raise ConnectionPoolError("数据库连接池异常")
    
    # 线程内持久化存储
    try:
        checkpointer = PostgresSaver(conn_pool)
        checkpointer.setup()
    except Exception as e:
        logger.error(f"检查点保存异常: {e}")
        raise ConnectionPoolError("检查点保存异常")
    
    # 跨线程持久化存储 
    try:
        store = PostgresStore(conn_pool, index={"dims": 1024, "embed": embed})
        store.setup()
    except Exception as e:
        logger.error(f"数据存储异常: {e}")
        raise ConnectionPoolError("数据存储异常")
    
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", lambda state, config: agent(state, config, store=store, llm=llm,tool_config=tool_config))
    workflow.add_node("call_tools", ParallelToolNode(tool_config.get_tools(), max_workers=5))
    workflow.add_node("rewrite", lambda state: rewrite(state, llm=llm))
    workflow.add_node("generate", lambda state: generate(state, llm=llm))
    workflow.add_node("grade_documents", lambda state: grade_documents(state, llm=llm))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", 
        tools_condition,
        {
            "tools": "call_tools",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "call_tools", 
        lambda state: route_after_tools(state, tool_config),
        {
            "generate": "generate",
            "grade_documents": "grade_documents"
        }
    )
    workflow.add_conditional_edges(
        "grade_documents", 
        route_after_grade,
        {
            "rewrite": "rewrite",
            "generate": "generate"
        }
    )
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=checkpointer, store=store)

def graph_response(graph: StateGraph, user_input: str, config: dict, tool_config: ToolConfig) -> None:
    """处理用户输入并输出响应，区分工具输出和大模型输出，支持多工具。

    Args:
        graph: 状态图实例。
        user_input: 用户输入。
        config: 运行时配置。
    """
    try:
        # 启动状态图流处理用户输入
        events = graph.stream({"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}, config)
        # 遍历事件流
        for event in events:
            # 遍历事件中的值
            for value in event.values():
                # 检查是否有有效消息
                if "messages" not in value or not isinstance(value["messages"], list):
                    logger.warning("No valid messages in response")
                    continue

                # 获取最后一条消息
                last_message = value["messages"][-1]

                # 检查消息是否包含工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    # 遍历工具调用
                    for tool_call in last_message.tool_calls:
                        # 检查工具调用是否为字典且包含名称
                        if isinstance(tool_call, dict) and "name" in tool_call:
                            # 记录工具调用日志
                            logger.info(f"Calling tool: {tool_call['name']}")
                    # 跳过本次循环
                    continue

                # 检查消息是否有内容
                if hasattr(last_message, "content"):
                    content = last_message.content

                    # 情况1：工具输出（动态检查工具名称）
                    if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
                        tool_name = last_message.name
                        print(f"Tool Output [{tool_name}]: {content}")
                    # 情况2：大模型输出（非工具消息）
                    else:
                        print(f"Assistant: {content}")
                else:
                    # 如果消息没有内容，可能是中间状态
                    logger.info("Message has no content, skipping")
                    print("Assistant: 未获取到相关回复")
    except ValueError as ve:
        logger.error(f"Value error in response processing: {ve}")
        print("Assistant: 处理响应时发生值错误")
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        print("Assistant: 处理响应时发生未知错误")

def main():
    conn_pool = None
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

        # 打印机器人就绪提示
        print("聊天机器人准备就绪！输入 'quit'、'exit' 或 'q' 结束对话。")
        # 定义运行时配置，包含线程ID和用户ID
        config = {"configurable": {"thread_id": "330", "user_id": "330"}}
        # 进入主循环
        while True:
            # 获取用户输入并去除首尾空格
            user_input = input("User: ").strip()
            # 检查是否退出
            if user_input.lower() in {"quit", "exit", "q"}:
                print("拜拜!")
                break
            # 检查输入是否为空
            if not user_input:
                print("请输入聊天内容！")
                continue
            # 处理用户输入并选择是否流式输出响应
            graph_response(graph, user_input, config, tool_config)

    except ConnectionPoolError as e:
        # 捕获连接池相关的异常
        logger.error(f"Connection pool error: {e}")
        print(f"错误: 数据库连接池问题 - {e}")
        sys.exit(1)
    except RuntimeError as e:
        # 捕获其他运行时错误
        logger.error(f"Initialization error: {e}")
        print(f"错误: 初始化失败 - {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        # 捕获键盘中断
        print("\n被用户打断。再见！")
    except Exception as e:
        # 捕获未预期的其他异常
        logger.error(f"Unexpected error: {e}")
        print(f"错误: 发生未知错误 - {e}")
        sys.exit(1)
    finally:
        # 清理资源
        if conn_pool and not conn_pool.closed:
            conn_pool.close()
            logger.info("Database connection pool closed")

if __name__ == "__main__":
    # 调用主函数
    main()