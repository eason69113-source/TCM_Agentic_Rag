import os 
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain.tools import tool

def get_tools(embed, llm):
    # 【核心修改 1】在这里预加载数据库，只加载一次，放入内存
    print("正在初始化工具：加载向量数据库...")
    try:
        db_path = "faiss_db"  # 如果你的 faiss_db 在别的地方，改成正确相对路径，比如 "../faiss_db"

        raw_path = os.path.join(db_path, "raw")
        trans_path = os.path.join(db_path, "trans")
        note_path = os.path.join(db_path, "note")
        vector_raw = FAISS.load_local(
            folder_path=raw_path,
            embeddings=embed,
            allow_dangerous_deserialization=True
        )
        vector_trans = FAISS.load_local(
            folder_path=trans_path,
            embeddings=embed,
            allow_dangerous_deserialization=True
        )
        vector_note = FAISS.load_local(
            folder_path=note_path,
            embeddings=embed,
            allow_dangerous_deserialization=True
        )
        print("向量数据库加载成功！")
        print(f"  原文向量数: {len(vector_raw.docstore._dict) if vector_raw else 0}")
        print(f"  直译向量数: {len(vector_trans.docstore._dict) if vector_trans else 0}")
        print(f"  解要向量数: {len(vector_note.docstore._dict) if vector_note else 0}")
    except Exception as e:
        print(f"警告：向量数据库加载失败，文档查询查询功能将不可用。错误: {e}")
        vector_raw = None
        vector_trans = None
        vector_note = None

    @tool('retriever_tool', parse_docstring=True)
    def retriever_tool(query: str):
        """这是《黄帝外经》查询工具。搜索并返回有关《黄帝外经》书籍中原文、直译、解要内容的信息。

        Args:
            query: 用户查询的问题

        Returns:
            返回在数据库中搜索到的与查询最相似的3个文档。
        """
        if vector_raw is None or vector_trans is None or vector_note is None:
            return "错误：《黄帝外经》数据库未成功加载，无法进行查询。"

        try:
            retrievers = []
            retrievers.append(vector_raw.as_retriever(search_kwargs={"k": 3}))
            retrievers.append(vector_trans.as_retriever(search_kwargs={"k": 3}))
            retrievers.append(vector_note.as_retriever(search_kwargs={"k": 3}))
            if not retrievers:
                return "错误：《黄帝外经》数据库未成功加载，无法进行查询。"
            weights = [0.2, 0.5, 0.3]
            retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
            docs = retriever.invoke(query)

            context_parts = []
            for doc in docs:
                meta = doc.metadata
                source = f"【{meta['篇名']} - {meta['字段']} 第{meta['段号']}段】"
                context_parts.append(f"{source}\n{doc.page_content.strip()}")
            
            return "\n\n".join(context_parts)

        except Exception as e:
            return f"检索过程发生错误: {e}"

    @tool('my_web_search1', parse_docstring=True)
    def web_search(query: str) -> str:
        """互联网搜索工具，可以搜索所有公开信息。

        Args:
            query: 需要进行互联网搜索的问题

        Returns:
            返回搜索的结果信息，是文本字符串。
        """
        try:
            # 【注意】请确保传入的 llm 对象里真的有 web_search 属性
            # 如果没有，这里建议直接使用 LangChain 的 DuckDuckGoSearchRun 或 Tavily
            if not hasattr(llm, 'web_search'):
                return "配置错误：传入的 LLM 对象不包含搜索模块。"
                
            res = llm.web_search.web_search(
                search_engine="search_pro",
                search_query=query,
            )
            # 增加空值校验
            if hasattr(res, 'search_result') and res.search_result:
                # 假设 search_result 是对象列表，取 content
                return "\n\n".join([str(d.content) for d in res.search_result])
            else:
                return "未搜索到相关结果。"
        except Exception as e:
            print(f"搜索错误: {e}")
            return f"搜索工具调用失败: {e}"
    
    return [retriever_tool, web_search]

class ToolConfig:
    def __init__(self, embed, llm):
        self.tools = get_tools(embed, llm)
        self.tool_names = {tool.name for tool in self.tools}
        self.tool_routing_config = self._build_routing_config(self.tools)
        
    def _build_routing_config(self, tools):
        routing_config = {}
        for tool in tools:
            tool_name = tool.name.lower() # 这里的 name 实际上是 'retriever_tool'

            if "retriev" in tool_name:
                routing_config[tool.name] = "grade_documents"
            else:
                routing_config[tool.name] = "generate"

        return routing_config
    
    def get_tools(self):
        return self.tools

    def get_tool_names(self):
        return self.tool_names

    def get_tool_routing_config(self):
        return self.tool_routing_config