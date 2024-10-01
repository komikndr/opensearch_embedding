import pytest
from opensearchpy import OpenSearch
from langchain_community.embeddings.opensearch import OpenSearchEmbeddings


@pytest.fixture
def model_id() -> str:
    """Fixture to provide model ID."""
    return "some-model-id"


@pytest.fixture
def client() -> OpenSearch:
    """Fixture to provide OpenSearch client connection."""
    return OpenSearch(
        hosts=[{'host': "localhost", 'port': 9200}],  # Remove sensitive info
        http_auth=("username", "password"),  # Remove sensitive info
        use_ssl=True,
        verify_certs=False
    )


@pytest.fixture
def opensearch_embedding(client, model_id) -> OpenSearchEmbeddings:
    return OpenSearchEmbeddings.from_connection(client, model_id)


def test_opensearch_embedding_documents(
        opensearch_embedding: OpenSearchEmbeddings) -> None:
    """
    Test OpenSearch embedding documents.
    Convert a list of strings into a list of floats,
    with each element having the shape of its embedding vector dimensions.
    """
    documents = ["foo bar", "bar foo", "foo"]
    output = opensearch_embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 768  # Expected embedding size
    assert len(output[1]) == 768  # Expected embedding size
    assert len(output[2]) == 768  # Expected embedding size


def test_opensearch_embedding_query(
        opensearch_embedding: OpenSearchEmbeddings) -> None:
    """
    Test OpenSearch embedding query.
    Convert a string into a float array, with the shape
    corresponding to its embedding vector dimensions.
    """
    document = "foo bar"
    output = opensearch_embedding.embed_query(document)
    assert len(output) == 768  # Expected embedding size

