from typing import List
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import torch


class E5LargeEmbeddings(Embeddings):
    """Multilingual E5-large embeddings implementation for Langchain"""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", device: str = None):
        """
        Initialize multilingual E5-large embeddings

        Args:
            model_name: Hugging Face model name for multilingual E5-large
                       (default: "intfloat/multilingual-e5-large")
            device: Device to run the model on ('cpu' or 'cuda').
                    If None, will use CUDA if available, otherwise CPU.
        """
        self.model_name = model_name

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading multilingual E5-large model ({self.model_name}) on {self.device}...")

        # Load the model with default pooling mode 'mean'
        self.model = SentenceTransformer(
            model_name,
            device=str(self.device),
            trust_remote_code=True
        )

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded successfully! Embedding dimension: {self.embedding_dim}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self.embedding_dim

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method to embed a list of texts
        """
        # Normalize texts by adding instruction prefixes
        normalized_texts = []
        for text in texts:
            # For E5 models, add instruction prefix for better performance
            if text.startswith("query:"):
                normalized_text = text  # Already has instruction
            else:
                normalized_text = f"query: {text}"

            # Ensure text is not empty
            if not text.strip():
                normalized_text = "passage: none"

            normalized_texts.append(normalized_text)

        # Encode texts with proper settings
        embeddings = self.model.encode(
            normalized_texts,
            batch_size=32,  # Adjust batch size based on GPU memory
            show_progress_bar=False,  # Disable progress bar for cleaner output
            convert_to_numpy=True,
            normalize_embeddings=True  # Optional: normalize embeddings
        )

        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents

        Args:
            texts: List of text documents to embed

        Returns:
            List of embeddings for each document
        """
        if not texts:
            return []

        try:
            return self._embed_texts(texts)
        except Exception as e:
            print(f"Error during document embedding: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * self.embedding_dim] * len(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text

        Args:
            text: Single text to embed

        Returns:
            Embedding for the text
        """
        if not text:
            return [0.0] * self.embedding_dim

        try:
            # Add query instruction for better search performance
            if not text.startswith("query:"):
                text = f"query: {text}"

            embedding = self.model.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            return embedding[0].tolist()
        except Exception as e:
            print(f"Error during query embedding: {str(e)}")
            return [0.0] * self.embedding_dim