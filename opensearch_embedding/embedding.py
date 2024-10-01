
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
#from langchain_core._api import deprecated
#from langchain_core.utils import get_from_env

if TYPE_CHECKING:
    from opensearchpy import OpenSearch

import json
from langchain_core.embeddings import Embeddings

class OpenSearchEmbedding(Embeddings):
    """
    A class to handle embedding generation using opensearchpy client for text embeddings.

    Attributes:
        client (OpenSearch): The OpenSearch client to connect to the OpenSearch instance.
        model_id (str): The model ID of the OpenSearch ML model.
                        https://host:9220/_plugins/_ml/_predict/text_embedding/MODEL_ID"
        input_field (str): The field where the input text is stored (default: 'text_field').
    """

    def __init__(
        self,
        client: OpenSearch,
        model_id: str,
        *,
        input_field: str = "text_field",
    ):
        self.client = client
        self.model_id = model_id
        self.input_field = input_field

    @classmethod
    def from_opensearch_connection(
        cls,
        opensearch_connection: OpenSearch,
        model_id: str,
        input_field: str = "text_field",
    ) -> OpenSearchEmbedding:
        """
        Class method to create an OpenSearchEmbedding object from an OpenSearch connection.

        Args:
            opensearch_connection (OpenSearch): The OpenSearch connection.
            model_id (str): The ML model ID for generating embeddings.
            input_field (str, optional): The input field for the text (default: 'text_field').

        Returns:
            OpenSearchEmbedding: An instance of the OpenSearchEmbedding class.
        """
        return cls(opensearch_connection, model_id, input_field=input_field)

    def _embedding_func(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method that sends a request to OpenSearch's text embedding endpoint
        and retrieves embeddings for the provided texts.

        Args:
            texts (List[str]): A list of strings to be embedded.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        endpoint = f"/_plugins/_ml/_predict/text_embedding/{self.model_id}"
        body = {
            "text_docs": texts,
            "return_number": True,
            "target_response": ["sentence_embedding"]
        }

        response = self.client.transport.perform_request(
            method="POST",
            url=endpoint,
            body=json.dumps(body),
        )
        # Extract embeddings from the response
        embeddings = [item['output'][0]['data'] for item in response['inference_results']]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts (List[str]): A list of text documents to embed.

        Returns:
            List[List[float]]: A list of embeddings for each document.
        """
        return self._embedding_func(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query.

        Args:
            text (str): The text query to embed.

        Returns:
            List[float]: The embedding for the query.
        """
        return self._embedding_func([text])[0]

       return self._embedding_func([text])[0]
