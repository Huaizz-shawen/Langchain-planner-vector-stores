# vector_stores/document_processor.py
import os
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
import json

# # Import Qwen embeddings if available
# try:
#     from .qwen_embeddings import QwenEmbeddings
#     qwen_available = True
# except ImportError:
#     try:
#         from qwen_embeddings import QwenEmbeddings
#         qwen_available = True
#     except ImportError:
#         QwenEmbeddings = None
#         qwen_available = False
#         print("Warning: Qwen embeddings not available. Install dashscope to use Qwen embeddings.")

# Import E5 embeddings if available
try:
    from .e5_embeddings import E5LargeEmbeddings
    e5_available = True
except ImportError:
    try:
        from e5_embeddings import E5LargeEmbeddings
        e5_available = True
    except ImportError:
        E5LargeEmbeddings = None
        e5_available = False
        print("Warning: E5 embeddings not available. Install sentence-transformers to use E5 embeddings.")

class DocumentVectorizer:
    """文档向量化处理器"""
    
    def __init__(
        self,
        embedding_model: str = "openai",  # 'openai' 或 'huggingface'
        vector_store_type: str = "chroma",  # 'chroma' 或 'faiss'
        persist_directory: str = "./vector_db"
    ):
        self.embedding_model_name = embedding_model
        self.vector_store_type = vector_store_type
        self.persist_directory = persist_directory
        
        # 初始化嵌入模型
        self.embeddings = self._init_embeddings()
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # 每个块的大小
            chunk_overlap=50,      # 块之间的重叠
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]
        )
    
    def _init_embeddings(self):
        """初始化嵌入模型"""
        if self.embedding_model_name == "openai":
            # 使用OpenAI的嵌入模型
            return OpenAIEmbeddings(
                model="text-embedding-3-small"  # 或 text-embedding-3-large
            )
        elif self.embedding_model_name == "huggingface":
            # 使用开源的中文嵌入模型（免费，本地运行）
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",  # 中文模型
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        elif self.embedding_model_name == "qwen":
            # 使用Qwen的嵌入模型
            if QwenEmbeddings is None:
                raise ImportError("Qwen embeddings not available. Please install dashscope: pip install dashscope")
            return QwenEmbeddings()
        elif self.embedding_model_name == "e5":
            # 使用多语言E5-large嵌入模型
            if E5LargeEmbeddings is None:
                raise ImportError("E5 embeddings not available. Please install sentence-transformers: pip install sentence-transformers")
            return E5LargeEmbeddings(
                model_name="intfloat/multilingual-e5-large"
            )
        else:
            raise ValueError(f"不支持的嵌入模型: {self.embedding_model_name}")
    
    def load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")
            
            documents = loader.load()
            print(f"✅ 成功加载文档: {file_path}")
            print(f"   文档数量: {len(documents)}")
            return documents
            
        except Exception as e:
            print(f"❌ 加载文档失败: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档切分成小块"""
        splits = self.text_splitter.split_documents(documents)
        print(f"📄 文档已切分为 {len(splits)} 个块")
        
        # 打印示例
        if splits:
            print(f"\n示例块（前100字符）:")
            print(f"  {splits[0].page_content[:100]}...")
        
        return splits
    
    def create_vector_store(
        self, 
        documents: List[Document],
        collection_name: Optional[str] = None
    ):
        """创建向量存储"""
        if not documents:
            raise ValueError("文档列表为空，无法创建向量存储")
        
        print(f"\n🔄 开始向量化...")
        print(f"   嵌入模型: {self.embedding_model_name}")
        print(f"   向量数据库: {self.vector_store_type}")
        
        if self.vector_store_type == "chroma":
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name or "default_collection"
            )
            vectorstore.persist()
            print(f"✅ 向量存储已保存到: {self.persist_directory}")
            
        elif self.vector_store_type == "faiss":
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # FAISS需要手动保存
            vectorstore.save_local(self.persist_directory)
            print(f"✅ FAISS索引已保存到: {self.persist_directory}")
        
        else:
            raise ValueError(f"不支持的向量存储类型: {self.vector_store_type}")
        
        return vectorstore
    
    def load_vector_store(self, collection_name: Optional[str] = None):
        """加载已存在的向量存储"""
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"向量存储目录不存在: {self.persist_directory}")
        
        print(f"📂 加载向量存储: {self.persist_directory}")
        
        if self.vector_store_type == "chroma":
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name or "default_collection"
            )
        elif self.vector_store_type == "faiss":
            vectorstore = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True  # FAISS需要这个参数
            )
        else:
            raise ValueError(f"不支持的向量存储类型: {self.vector_store_type}")
        
        print("✅ 向量存储加载成功")
        return vectorstore
    
    def process_file(
        self, 
        file_path: str,
        collection_name: Optional[str] = None
    ):
        """完整的文件处理流程：加载 → 切分 → 向量化"""
        print(f"\n{'='*60}")
        print(f"开始处理文件: {file_path}")
        print(f"{'='*60}\n")
        
        # 1. 加载文档
        documents = self.load_document(file_path)
        if not documents:
            return None
        
        # 2. 切分文档
        splits = self.split_documents(documents)
        
        # 3. 创建向量存储
        vectorstore = self.create_vector_store(splits, collection_name)
        
        print(f"\n{'='*60}")
        print(f"✅ 处理完成！")
        print(f"{'='*60}\n")
        
        return vectorstore
    
    def add_documents_to_existing(
        self,
        file_path: str,
        collection_name: Optional[str] = None
    ):
        """向已有的向量存储添加新文档"""
        # 加载现有向量存储
        vectorstore = self.load_vector_store(collection_name)
        
        # 加载和切分新文档
        documents = self.load_document(file_path)
        if not documents:
            return vectorstore
        
        splits = self.split_documents(documents)
        
        # 添加到现有向量存储
        print(f"\n🔄 添加 {len(splits)} 个文档块到现有向量存储...")
        vectorstore.add_documents(splits)
        
        if self.vector_store_type == "chroma":
            vectorstore.persist()
        elif self.vector_store_type == "faiss":
            vectorstore.save_local(self.persist_directory)
        
        print("✅ 文档已添加到向量存储")
        return vectorstore