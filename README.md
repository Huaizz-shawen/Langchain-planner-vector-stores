# Qwen Embeddings for Document Vectorization

This implementation provides support for using Qwen (Tongyi) embeddings in place of OpenAI embeddings for document vectorization tasks. The system supports multiple embedding models including Qwen, HuggingFace, and OpenAI.

## Features

- **Qwen Embeddings Support**: Use Tongyi Qwen embeddings instead of OpenAI
- **Multiple Embedding Options**: Qwen, HuggingFace, and OpenAI embeddings
- **File Support**: TXT, PDF, DOCX, DOC formats
- **Vector Stores**: Chroma and FAISS support
- **Easy Integration**: Drop-in replacement for existing code

## Setup

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install additional dependencies if needed:
   ```bash
   pip install dashscope
   ```

## Usage

### Setting up Qwen API Key

```bash
export DASHSCOPE_API_KEY="your-qwen-api-key-here"
```

### Basic Example

```python
from vector_stores.document_processor import DocumentVectorizer

# Initialize with Qwen embeddings
vectorizer = DocumentVectorizer(
    embedding_model="qwen",  # Use Qwen embeddings
    vector_store_type="chroma",  # or "faiss"
    persist_directory="./my_vector_db"
)

# Process a document
vectorstore = vectorizer.process_file(
    file_path="./data/my_document.txt",
    collection_name="my_knowledge_base"
)

# Perform similarity search
query = "your search query"
results = vectorstore.similarity_search(query, k=3)
```

### Available Embedding Models

- `"qwen"` - Qwen embeddings (requires DASHSCOPE_API_KEY)
- `"huggingface"` - Local HuggingFace embeddings (free, no API key needed)
- `"openai"` - OpenAI embeddings (requires OPENAI_API_KEY)

## Architecture

```
Document → Text Splitter → Embeddings → Vector Store
```

The system uses:
- `RecursiveCharacterTextSplitter` to chunk documents
- Different embedding models for vectorization
- Chroma or FAISS for vector storage

## Testing

Run the test script to verify all embedding types:

```bash
python -m vector_stores.test_qwen_embeddings
```

## Troubleshooting

- **Import Errors**: Make sure you have installed all required packages
- **API Key Errors**: Ensure DASHSCOPE_API_KEY environment variable is set for Qwen
- **Missing Dependencies**: Install missing packages as prompted in error messages

## Notes

- Qwen embeddings require an active API key from Tongyi
- HuggingFace embeddings are free and run locally
- The system preserves all original functionality while adding Qwen support
- All embedding models are drop-in replacements for each other