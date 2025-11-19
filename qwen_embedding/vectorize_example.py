# vector_stores/vectorize_example.py
from document_processor import DocumentVectorizer
import os

# 方式1: 使用Qwen嵌入模型（需要API key）
def example_qwen():
    # 设置Qwen API密钥
    os.environ["DASHSCOPE_API_KEY"] = "sk-f0a76440f58d41399c874fb394e141ce"

    # 初始化向量化器
    vectorizer = DocumentVectorizer(
        embedding_model="qwen",  # 使用qwen嵌入模型
        vector_store_type="chroma",
        persist_directory="./vector_db"
    )

    # 处理单个文件
    vectorstore = vectorizer.process_file(
        file_path="./data/sii_introduction_withaction.txt",
        collection_name="sii_introduction_base"
    )

    # 测试检索
    query = "请介绍一下上海创智学院？"
    results = vectorstore.similarity_search(query, k=3)

    print(f"\n查询: {query}")
    print(f"找到 {len(results)} 个相关结果:\n")
    for i, doc in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  {doc.page_content[:200]}...")
        print()

# 方式2: 使用OpenAI嵌入模型（需要API key）
def example_openai():
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"

    # 初始化向量化器
    vectorizer = DocumentVectorizer(
        embedding_model="openai",
        vector_store_type="chroma",
        persist_directory="./my_vector_db"
    )

    # 处理单个文件
    vectorstore = vectorizer.process_file(
        file_path="./data/my_document.txt",
        collection_name="my_knowledge_base"
    )

    # 测试检索
    query = "如何使用LangChain？"
    results = vectorstore.similarity_search(query, k=3)

    print(f"\n查询: {query}")
    print(f"找到 {len(results)} 个相关结果:\n")
    for i, doc in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  {doc.page_content[:200]}...")
        print()

# 方式3: 使用HuggingFace本地嵌入模型（免费，无需API key）
def example_huggingface():
    # 初始化向量化器
    vectorizer = DocumentVectorizer(
        embedding_model="huggingface",  # 使用HuggingFace嵌入模型
        vector_store_type="chroma",
        persist_directory="./my_vector_db"
    )

    # 处理单个文件
    vectorstore = vectorizer.process_file(
        file_path="./data/my_document.txt",
        collection_name="my_knowledge_base"
    )

    # 测试检索
    query = "如何使用LangChain？"
    results = vectorstore.similarity_search(query, k=3)

    print(f"\n查询: {query}")
    print(f"找到 {len(results)} 个相关结果:\n")
    for i, doc in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  {doc.page_content[:200]}...")
        print()

if __name__ == "__main__":
    example_qwen()