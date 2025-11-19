
E5-large Multilingual Embeddings Quick Start Guide

This setup enables you to use the multilingual E5-large embedding model
for RAG (Retrieval-Augmented Generation) applications.

**Features:**
- Supports 100+ languages including Chinese, English, German, French, etc.
- 1024-dimensional embeddings for high-quality semantic search
- No API key required (runs locally)
- Optimized for retrieval tasks with instruction prefixes

**Installation:**

1. For CPU-only environments:
   ```bash
   pip install torch sentence-transformers langchain langchain-community chromadb
   ```

2. For GPU support (CUDA):
   ```bash
   # CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   pip install sentence-transformers langchain langchain-community chromadb
   ```

3. Using conda:
   ```bash
   conda env create -f conda_environment.yml
   conda activate sentence-transformers
   ```

**Quick Start:**

1. Basic Usage:
   ```python
   from document_processor import DocumentVectorizer

   # Initialize with E5-large
   vectorizer = DocumentVectorizer(
       embedding_model="e5",
       vector_store_type="chroma",
       persist_directory="./vector_db_e5"
   )

   # Process documents
   vectorstore = vectorizer.process_file(
       file_path="../data/your_document.txt",
       collection_name="sii_collection"
   )
   ```

2. Test Retrieval:
   ```python
   # Search in multiple languages
   queries = [
       "What is the main topic?",     # English
       "主なトピックは何ですか？",      # Japanese
       "Was sind die Hauptthemen?",   # German
   ]

   for query in queries:
       results = vectorstore.similarity_search(query, k=3)
       print(f"Results for '{query}': {len(results)} found")
   ```

3. Direct Embedding Usage:
   ```python
   from e5_embeddings import E5LargeEmbeddings

   # Initialize
   e5 = E5LargeEmbeddings()

   # Embed documents
   texts = ["Your text here", "Another document"]
   embeddings = e5.embed_documents(texts)

   # Embed query
   query = "Search query"
   query_embedding = e5.embed_query(query)
   ```

**Device Management:**
- CPU: `E5LargeEmbeddings(device="cpu")`
- GPU: `E5LargeEmbeddings(device="cuda")` (if available)
- Auto-detect: `E5LargeEmbeddings()` (prefers CUDA if available)

**Example Scripts:**
- `example_e5.py` - Complete usage examples
- `vectorize_example.py` - General vectorization examples

**Model Information:**
- Model: `intfloat/multilingual-e5-large`
- Embedding dimension: 1024
- Supported languages: 100+
- Average inference time: ~50-100ms per document (CPU)
- Model size: ~560MB (download required first time)

**Performance Tips:**
1. Use batch processing for multiple documents
2. Enable GPU for faster inference on large datasets
3. Adjust chunk_size based on your documents
4. Consider normalize_embeddings=True for better performance

**Error Handling:**
- If model download fails, check internet connection
- For CUDA out of memory, use device="cpu"
- First run downloads model (~560MB) - be patient

**Files Added:**
- `e5_embeddings.py` - Core E5-large embedding wrapper
- `example_e5.py` - Complete usage examples
- `requirements_e5.txt` - Package dependencies
- `e5_guide.md` - This guide
