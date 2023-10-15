import os
from flask import Flask, render_template, request, Response
from waitress import serve

from llama_index import StorageContext, load_index_from_storage
from llama_index.prompts import Prompt
from llama_index.vector_stores import RedisVectorStore
from llama_index import VectorStoreIndex
from models.mistral import service_context
from llama_index.embeddings import HuggingFaceEmbedding

import sys

db = sys.argv[1]

text_qa_template_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using both the context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
    "If the context isn't helpful, you can also answer the question on your own.\n"
)
text_qa_template = Prompt(text_qa_template_str)

refine_template_str = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Using both the new context and your own knowledege, update or repeat the existing answer.\n"
)
refine_template = Prompt(refine_template_str)

# get path for static files
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    static_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'static')


# start server
print("\033[96mStarting Squall at http://127.0.0.1:1333\033[0m")
squall = Flask(__name__, static_folder=static_dir, template_folder=static_dir)

# server landing page
@squall.route('/')
def landing():
    return render_template('index.html')

# run and stream the response
@squall.route('/run', methods=['POST'])
def run():
    data = request.json

    vector_store = RedisVectorStore(
        index_name=db,
        index_prefix=db,
        redis_url="redis://localhost:6379",
        overwrite=True,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

    engine = index.as_query_engine(
        service_context=service_context,
        streaming=True,
        )
    
    def event_stream():
        response = engine.query(data['input'])
        for line in response.response_gen:
            yield line

    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    print("\033[93mSquall started. Press CTRL+C to quit.\033[0m")    
    serve(squall, port=1333, threads=4)
    # squall.run(port=1333,debug=True)
