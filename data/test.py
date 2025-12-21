# test_rag.py

import os
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from config import Config

# ====================== 加载数据库 ======================
# ====================== 加载数据库 ======================
db_path = "faiss_db"  # 如果你的 faiss_db 在别的地方，改成正确相对路径，比如 "../faiss_db"

raw_path = os.path.join(db_path, "raw")
trans_path = os.path.join(db_path, "trans")
note_path = os.path.join(db_path, "note")

raw_index = FAISS.load_local(raw_path, Config.embed1, allow_dangerous_deserialization=True) \
    if os.path.exists(raw_path) else None
trans_index = FAISS.load_local(trans_path, Config.embed1, allow_dangerous_deserialization=True) \
    if os.path.exists(trans_path) else None
note_index = FAISS.load_local(note_path, Config.embed1, allow_dangerous_deserialization=True) \
    if os.path.exists(note_path) else None

print("数据库加载完成！")
print(f"  原文向量数: {len(raw_index.docstore._dict) if raw_index else 0}")
print(f"  直译向量数: {len(trans_index.docstore._dict) if trans_index else 0}")
print(f"  解要向量数: {len(note_index.docstore._dict) if note_index else 0}")

# ====================== 创建融合检索器 ======================
def get_ensemble_retriever(k=6):
    retrievers = []
    if raw_index:
        retrievers.append(raw_index.as_retriever(search_kwargs={"k": k}))
    if trans_index:
        retrievers.append(trans_index.as_retriever(search_kwargs={"k": k}))
    if note_index:
        retrievers.append(note_index.as_retriever(search_kwargs={"k": k}))

    if not retrievers:
        return None

    # 权重：原文 0.2、直译 0.5、解要 0.3（你可以调，直译权重高点更准确易懂）
    weights = [0.2, 0.5, 0.3]
    return EnsembleRetriever(retrievers=retrievers, weights=weights)

retriever = get_ensemble_retriever(k=6)

# ====================== RAG 查询函数 ======================
def ask_question(query: str):
    if retriever is None:
        print("检索器未初始化！")
        return

    # 检索相关片段
    docs = retriever.invoke(query)

    # 构建带来源的上下文
    context_parts = []
    for doc in docs:
        meta = doc.metadata
        source = f"【{meta['篇名']} - {meta['字段']} 第{meta['段号']}段】"
        context_parts.append(f"{source}\n{doc.page_content.strip()}")
    
    context = "\n\n".join(context_parts)

    # Prompt（严格要求依据资料回答）
    prompt = f"""你是一位精通《黄帝外经》的中医古籍专家，请根据以下提供的原文、直译、解要内容，严谨、准确地回答用户问题。
如果资料不足以回答，请直接回复“根据当前《黄帝外经》资料无法确定”。

资料：
{context}

问题：{query}

请用现代中文、自然流畅地回答："""

    print("\n" + "="*60)
    print("检索到的相关内容：")
    print(context)
    print("="*60)
    print("AI 回答：")
    
    # 调用 LLM（支持流式输出）
    response = Config.llm1.invoke(prompt)
    print(response.content)

# ====================== 测试几个问题 ======================
if __name__ == "__main__":
    print("\n开始测试《黄帝外经》RAG 系统\n")
    
    questions = [
        "上古之人如何养生？"
    ]

    for q in questions:
        print(f"\n问题：{q}")
        ask_question(q)
        print("\n" + "-"*80)
    
    # 最后让你自己输入问题测试
    print("\n测试完成！现在你可以自己输入问题继续测试（输入 exit 退出）")
    while True:
        user_q = input("\n你的问题：").strip()
        if user_q.lower() in ["exit", "quit", "退出"]:
            break
        if user_q:
            ask_question(user_q)