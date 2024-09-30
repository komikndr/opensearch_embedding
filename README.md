# OpenSearch Embedding Library

This is a Python library for generating text embeddings using OpenSearch embedding ML plugin for langchain.
It allows you to easily create document and query embeddings using the OpenSearch client.

- Embed documents and queries using OpenSearch's ML plugin.

## Installation

To install the library, first clone the repository and then install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage
```python
from opensearchpy import OpenSearch
from opensearch_embedding import OpenSearchEmbedding

# Connect to OpenSearch
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_auth=('admin', 'admin'),
)

#For Insecure Non SSL
client = OpenSearch(
    hosts=[{'host': "localhost", 'port': 9200}],
    http_auth=("admin", "admin"),
    use_ssl=True,
    verify_certs=False
)


#The model id can be obtained by https://host:9220/_plugins/_ml/_predict/text_embedding/MODEL_ID"
embedding_model = OpenSearchEmbedding(client, model_id='my-model-id')

documents = ["This is a sample document.", "Another document for embedding."]
embeddings = embedding_model.embed_documents(documents)
print(embeddings)

query = "Sample query"
query_embedding = embedding_model.embed_query(query)
print(query_embedding)
```


