import os

from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = HuggingFaceEmbeddings()


api_base = os.environ.get("OPENAI_API_BASE")
api_key = os.environ.get("OPENAI_API_KEY")
api_type = os.environ.get("OPENAI_API_TYPE")
api_version = os.environ.get("OPENAI_API_VERSION")

# [print(x) for x in [api_base, api_key, api_type, api_version]]



embeddings = embedding_model.embed_documents(
    [
        "The Trailblazers beat the Lakers in the basketball game last night.",
        "Our game idea is simple: bake a cake without eggs.",
        "Who knew the biggest video game this century would be a pixel side scroller?"
    ],
)

query_embedded = embedding_model.embed_query("what's the most popular video game?")

for doc in range(len(embeddings)):
    print(f"similarity between query and doc{doc}:{cosine_similarity([query_embedded], [embeddings[doc]])} ")
