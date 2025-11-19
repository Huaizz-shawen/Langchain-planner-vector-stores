"""
Example script for using the multilingual E5-large embedding model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from document_processor import DocumentVectorizer
import torch

def example_e5():
    """Example of using E5-large multilingual embeddings"""

    print("=" * 60)
    print("Testing E5-large Multilingual Embeddings")
    print("=" * 60)

    # Initialize the vectorizer with E5-large embeddings
    vectorizer = DocumentVectorizer(
        embedding_model="e5",  # Use E5-large embedding model
        vector_store_type="chroma",  # Use chroma vector store
        persist_directory="./vector_db_e5"  # Directory for vector store
    )

    print("\n✅ E5-large vectorizer initialized successfully!")

    # Process a single file
    file_path = "../data/sii_introduction_withaction.txt"  # Replace with your file path

    if os.path.exists(file_path):
        # Create vector store from document
        vectorstore = vectorizer.process_file(
            file_path=file_path,
            collection_name="sii_e5_collection"
        )

        # Test retrieval with different queries
        print("\n🔥 Testing retrieval performance with different queries:\n")

        test_queries = [
            "请介绍一下上海创智学院？",  # Chinese query
            "What is Shanghai Innovation Institution?",  # English query
            "Was sind die Möglichkeiten für die Teilnahme am S-TAL-Plattform?",  # German query
            "Quelles sont les conditions d'admission?",  # French query
        ]

        for query in test_queries:
            print(f"Query: {query}")
            results = vectorstore.similarity_search(query, k=2)
            print(f"Found {len(results)} relevant results:")

            for i, doc in enumerate(results, 1):
                print(f"  Result {i}: {doc.page_content[:150]}...")
            print()

        print("=" * 60)
        print("✅ E5-large embedding test completed successfully!")
        print("=" * 60)

    # else:
    #     print(f"❌ File not found: {file_path}")
    #     print("Please create the data directory and place your documents there.\n")
    #     print("Creating a simple mock document to demonstrate E5-large functionality...")

    #     # Create a mock document directory
    #     os.makedirs("./data", exist_ok=True)

    #     # Create a simple test document
    #     mock_content = """This is a test document about Shanghai Innovation Institution.
    #     上海创智学院是一所创新的教育机构。
    #     L'institut d'Innovation de Shanghai est une institution éducative novatrice.
    #     Das Shanghaier Innovationsinstitut ist eine innovative Bildungseinrichtung.

    #     They specialize in AI education and advanced technology research.
    #     The institution offers various programs and courses.
    #     We provide opportunities in various fields.
    #     它为不同领域的创新和创业提供了平台

    #     Model: The E5 embedding model is multilingual and handles various languages effectively.
    #     这个模型支持多种语言，包括中文、英文等
    #     Le modèle prend en charge de nombreuses langues sans problème.
    #     """

    #     with open("./data/test_document.txt", "w", encoding="utf-8") as f:
    #         f.write(mock_content)

    #     print("✅ Mock document created: ./data/test_document.txt")
    #     print("Please run the example again to test with the mock document.")

def example_device_control():
    """Demonstrate how to control the device for E5 embeddings"""

    # Add parent directory to path to import e5_embeddings
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from e5_embeddings import E5LargeEmbeddings

    # Example: Use CPU
    print("\n=== Using CPU ===")
    cpu_embeddings = E5LargeEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        device="cpu"
    )

    print(f"✅ CPU embeddings initialized")

    # Example: Use CUDA (if available)
    print("\n=== Testing CUDA availability ===")
    try:
        gpu_embeddings = E5LargeEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            device="cuda"
        )
        print(f"✅ GPU embeddings initialized")
        gpu_type = "CUDA" if torch.cuda.is_available() else "CPU fallback"
        print(f"Device actually used: {gpu_type}")
    except Exception as e:
        print(f"⚠️  GPU not available: {e}")

    # Test embeddings
    texts = ["Hello world", "你好世界"]
    cpu_result = cpu_embeddings.embed_documents(texts)
    print(f"Embeddings shape: {len(cpu_result)} documents, {len(cpu_result[0])} dimensions")


def example_add_documents():
    """Demonstrate adding documents to an existing vector store"""

    print("=" * 60)
    print("Adding documents to existing vector store")
    print("=" * 60)

    vectorizer = DocumentVectorizer(
        embedding_model="e5",
        vector_store_type="chroma",
        persist_directory="./vector_db_e5"
    )

    # Add a new document to existing collection
    new_file_path = "../data/additional_document.txt"
    if os.path.exists(new_file_path):
        vectorstore = vectorizer.add_documents_to_existing(
            file_path=new_file_path,
            collection_name="sii_e5_collection"
        )
        print("✅ Document added successfully!")
    else:
        print(f"❌ File not found: {new_file_path}")


def example_instruct_queries():
    """Demonstrate the power of instruct-based queries with E5-large"""

    # Add parent directory to path to import e5_embeddings
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from e5_embeddings import E5LargeEmbeddings

    print("=" * 60)
    print("Testing E5-large instruct-based retrieval")
    print("=" * 60)

    # Initialize embeddings
    try:
        e5 = E5LargeEmbeddings(model_name="intfloat/multilingual-e5-large")

        # Test embedding simple queries and documents
        documents = [
            "Shanghai Innovation Institution provides AI education programs.",
            "The E5 model is a powerful multilingual embedding model."
        ]

        # Embed documents
        print("\n📄 Embedding documents...")
        doc_embeddings = e5.embed_documents(documents)
        print(f"Document embeddings created: {len(doc_embeddings)} documents")
        print(f"Embedding dimension: {len(doc_embeddings[0])}")

        # Embed a test query
        print("\n🎯 Testing query embedding...")
        test_query = "What is SII and what programs do they offer?"

        query_embed = e5.embed_query(test_query)
        print(f"Query embedding generated: {len(query_embed)} dimensions")

        print("✅ E5-large instruct-based retrieval test completed!")

    except Exception as e:
        print(f"⚠️  Error during E5 instruction test: {e}")
        return


if __name__ == "__main__":
    print("E5-Large Multilingual Embedding Examples\n")

    # Main example
    example_e5()

    # print("\n" + "=" * 60)
    # print("Additional examples:")
    # print("=" * 60)

    # # Run device control example
    # print("\n1. Device Control Example:")
    # try:
    #     example_device_control()
    # except Exception as e:
    #     print(f"⚠️  Device control example failed: {e}")

    # # Run document addition example
    # print("\n2. Document Addition Example:")
    # try:
    #     example_add_documents()
    # except Exception as e:
    #     print(f"⚠️  Document addition example failed: {e}")

    # # Run instruct queries example
    # print("\n3. Instruct Queries Example:")
    # try:
    #     example_instruct_queries()
    # except Exception as e:
    #     print(f"⚠️  Instruct queries example failed: {e}")