from llama_index import ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.embeddings import HuggingFaceEmbedding

llm = LlamaCPP(
    model_path="./llms/mistral-7b-v0.1.Q5_K_S.gguf",
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)