from typing import List, Optional
import numpy as np
from langchain.embeddings.base import Embeddings
import dashscope  # Qwen SDK
import os


class QwenEmbeddings(Embeddings):
    """Qwen embeddings implementation for Langchain"""
    
    def __init__(self, model: str = "text-embedding-v1", api_key: Optional[str] = None):
        """
        Initialize Qwen embeddings
        
        Args:
            model: Qwen embedding model name (default: text-embedding-v1)
            api_key: Qwen API key (if not provided, will use DASHSCOPE_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Qwen API key is required. Set DASHSCOPE_API_KEY environment variable or pass api_key parameter.")
        
        # Set the API key for dashscope
        dashscope.api_key = self.api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embeddings for each document
        """
        embeddings = []
        
        for text in texts:
            # Truncate text if too long (Qwen has limits)
            # For longer texts, you may need to chunk them
            truncated_text = text[:2000] if len(text) > 2000 else text
            
            try:
                response = dashscope.TextEmbedding.call(
                    model=self.model,
                    input=truncated_text
                )
                
                if response.status_code == 200:
                    embedding = response.output['embeddings'][0]['embedding']
                    embeddings.append(embedding)
                else:
                    # Handle error - return zeros or raise exception
                    print(f"Error from Qwen API: {response.code} - {response.message}")
                    # Return a zero vector as fallback
                    embeddings.append([0.0] * 1536)  # Assuming 1536-dim embedding
                    
            except Exception as e:
                print(f"Exception during embedding: {str(e)}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * 1536)
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        
        Args:
            text: Single text to embed
            
        Returns:
            Embedding for the text
        """
        # Truncate text if too long
        truncated_text = text[:2000] if len(text) > 2000 else text
        
        try:
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=truncated_text
            )
            
            if response.status_code == 200:
                embedding = response.output['embeddings'][0]['embedding']
                return embedding
            else:
                print(f"Error from Qwen API: {response.code} - {response.message}")
                return [0.0] * 1536  # Return zero vector as fallback
                
        except Exception as e:
            print(f"Exception during query embedding: {str(e)}")
            return [0.0] * 1536