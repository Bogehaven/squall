from llama_index import VectorStoreIndex, SimpleDirectoryReader
from models.mistral import service_context
from llama_index.vector_stores import RedisVectorStore
from llama_index.storage.storage_context import StorageContext


import sys

db = sys.argv[1]
documents = SimpleDirectoryReader(f"./data/{db}").load_data()

vector_store = RedisVectorStore(
    index_name=db,
    index_prefix=db,
    redis_url="redis://localhost:6379",
    overwrite=True,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)


index = VectorStoreIndex.from_documents(documents,
                                        storage_context=storage_context,
                                        service_context=service_context,
                                        show_progress=True)