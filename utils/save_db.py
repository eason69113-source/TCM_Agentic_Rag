# agent/vector_db.py  （建议新建一个文件专门管理向量库）

import uuid
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from ..config import Config
import json

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    chunk_size=800,
    chunk_overlap=120,
)

def field_to_docs(record: dict, field: str):
    title = record.get("篇名", "")
    text = (record.get(field) or "").strip()
    if not text:
        return []

    chunks = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks, start=1):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "id": str(uuid.uuid4()),
                "篇名": title,
                "字段": field,
                "段号": i
            }
        ))
    return docs

def build_and_save_db(records: list[dict], db_path: str = "faiss_db"):
    """
    一次性构建并保存三个独立的 FAISS 索引到本地文件夹
    db_path 下会生成三个子文件夹：raw / trans / note
    """
    os.makedirs(db_path, exist_ok=True)
    
    raw_docs, trans_docs, note_docs = [], [], []

    for r in records:
        raw_docs.extend(field_to_docs(r, "原文"))
        trans_docs.extend(field_to_docs(r, "廖冬晴直译"))
        note_docs.extend(field_to_docs(r, "梅自强解要"))

    # 构建并保存
    if raw_docs:
        raw_index = FAISS.from_documents(raw_docs, Config.embed1)
        raw_index.save_local(os.path.join(db_path, "raw"))
        print(f"原文索引保存完成，共 {len(raw_docs)} 条向量")
    
    if trans_docs:
        trans_index = FAISS.from_documents(trans_docs, Config.embed1)
        trans_index.save_local(os.path.join(db_path, "trans"))
        print(f"直译索引保存完成，共 {len(trans_docs)} 条向量")
    
    if note_docs:
        note_index = FAISS.from_documents(note_docs, Config.embed1)
        note_index.save_local(os.path.join(db_path, "note"))
        print(f"解要索引保存完成，共 {len(note_docs)} 条向量")

    print(f"所有向量数据库已保存至：{db_path}")

if __name__ == "__main__":
    with open("data/huangdi_waijing.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    build_and_save_db(records, db_path="faiss_db")