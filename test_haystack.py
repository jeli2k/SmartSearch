from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever

document_store = ElasticsearchDocumentStore(
    host="elasticsearch",  # name defined in your docker-compose.yml
    port=9200,
    username="",  # leave blank if not using security features
    password="",  # leave blank if not using security features
    index="documents"  # index name
)

# sample document
sample_doc = {
    "content": "This is a test document. If you see this, Haystack and Elasticsearch is working properly.",
    "meta": {"source": "test"}
}

# write the document to the store
document_store.write_documents([sample_doc])

# read documents back from the store
documents = document_store.get_all_documents()
for doc in documents:
    print(doc.content)

