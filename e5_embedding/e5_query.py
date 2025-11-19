"""
query only
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from document_processor import DocumentVectorizer

def query_e5():
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

    # Load the existing vector database
    vectorstore = vectorizer.load_vector_store(collection_name="sii_e5_collection")

    print("\n✅ E5-large vectorizer initialized and loaded successfully!")


    # Test retrieval with different queries and strategies
    print("\n🔥 Testing retrieval performance with different strategies:\n")

    test_queries = [
        "请介绍一下上海创智学院？",  # Chinese query
        "What is Shanghai Innovation Institution?",  # English query
        "上海创智学院的老师来自哪里？",  # Another Chinese query
        "上海创智学院的培养目标是什么？"   # Another Chinese query
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}\n")

        # Strategy 1: Standard similarity search (baseline)
        print("📍 Strategy 1: Standard Similarity Search (k=2)")
        results = vectorstore.similarity_search(query, k=2)
        for i, doc in enumerate(results, 1):
            print(f"  Result {i}: {doc.page_content[:150]}...")

        # Strategy 2: MMR for diversity
        print("\n📍 Strategy 2: MMR Search (k=4, fetch_k=20, lambda_mult=0.5)")
        print("  (Balances relevance vs diversity)")
        try:
            mmr_results = vectorstore.max_marginal_relevance_search(
                query,
                k=4,  # Return top 4 diverse results
                fetch_k=20,  # Fetch 20 candidates first
                lambda_mult=0.5  # 0.5 = balance relevance and diversity (0=max diversity, 1=max relevance)
            )
            for i, doc in enumerate(mmr_results, 1):
                print(f"  Result {i}: {doc.page_content[:150]}...")
        except Exception as e:
            print(f"  ❌ MMR not available: {e}")

        # Strategy 3: Larger k with re-ranking by score
        print("\n📍 Strategy 3: Re-ranking with Scores (fetch k=10, return top 4)")
        print("  (Retrieve more, then re-rank by similarity score)")
        try:
            results_with_scores = vectorstore.similarity_search_with_score(query, k=10)
            # Sort by score (lower is better for distance metrics)
            sorted_results = sorted(results_with_scores, key=lambda x: x[1])
            top_results = sorted_results[:4]

            for i, (doc, score) in enumerate(top_results, 1):
                print(f"  Result {i} (score: {score:.4f}): {doc.page_content[:150]}...")
        except Exception as e:
            print(f"  ❌ Scoring not available: {e}")

        print()

    print("=" * 60)
    print("✅ E5-large embedding query completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    query_e5()