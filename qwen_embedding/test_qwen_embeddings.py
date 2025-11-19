"""
Test script to verify Qwen embeddings implementation
"""
import os
from vector_stores.document_processor import DocumentVectorizer

def test_qwen_embeddings():
    """
    Test function to verify that Qwen embeddings work correctly
    """
    print("Testing Qwen embeddings implementation...")
    
    try:
        # Initialize vectorizer with Qwen embeddings
        vectorizer = DocumentVectorizer(
            embedding_model="qwen",  # Use Qwen embeddings
            vector_store_type="chroma",
            persist_directory="./test_vector_db"
        )
        
        print("✅ Qwen embeddings initialized successfully!")
        print(f"Embedding model: {vectorizer.embedding_model_name}")
        print(f"Vector store type: {vectorizer.vector_store_type}")
        
        # Test embedding a sample text
        sample_text = ["这是一个测试文本", "This is a test text"]
        embeddings = vectorizer.embeddings.embed_documents(sample_text)
        
        print(f"✅ Successfully generated embeddings for {len(sample_text)} texts")
        print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
        
        return True
        
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        print("This might be because dashscope is not installed.")
        print("Install it using: pip install dashscope")
        return False
        
    except ValueError as e:
        print(f"❌ ValueError: {e}")
        if "API key" in str(e):
            print("Please set your DASHSCOPE_API_KEY environment variable.")
            print("Example: export DASHSCOPE_API_KEY='your-api-key-here'")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_available_embedding_types():
    """
    Test which embedding types are available
    """
    print("\nTesting available embedding types...")
    
    # Test 1: HuggingFace (should always work if dependencies installed)
    try:
        vectorizer = DocumentVectorizer(
            embedding_model="huggingface",
            vector_store_type="chroma",
            persist_directory="./test_hf_db"
        )
        print("✅ HuggingFace embeddings available")
    except Exception as e:
        print(f"❌ HuggingFace embeddings error: {e}")
    
    # Test 2: Qwen
    try:
        vectorizer = DocumentVectorizer(
            embedding_model="qwen",
            vector_store_type="chroma",
            persist_directory="./test_qwen_db"
        )
        print("✅ Qwen embeddings available")
    except ImportError:
        print("❌ Qwen embeddings not available (dashscope not installed)")
    except Exception as e:
        print(f"⚠️  Qwen embeddings issue: {e}")
        
    # Test 3: OpenAI
    try:
        vectorizer = DocumentVectorizer(
            embedding_model="openai",
            vector_store_type="chroma",
            persist_directory="./test_openai_db"
        )
        print("✅ OpenAI embeddings available")
    except Exception as e:
        print(f"⚠️  OpenAI embeddings issue: {e}")

if __name__ == "__main__":
    print("="*60)
    print("Qwen Embeddings Test")
    print("="*60)
    
    # Test embedding types availability
    test_available_embedding_types()
    
    print("\n" + "="*60)
    
    # Test Qwen specifically
    success = test_qwen_embeddings()
    
    print("\n" + "="*60)
    if success:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed or were skipped.")
    print("="*60)