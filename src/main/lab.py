"""
    Creating text embeddings is the process of turning text into vectors (think array 
    of numbers that represent the text of the document). This is done using an embedding model
    that parses meaning from the text and creates a vector array to represent the meaning of the text.
    The benefit of using these embeddings comes from the fact that computers (and machine learning models)
    are able to work with numbers much easier than text, so embeddings are the more optimal way of interacting
    with your documents than trying to interact with the text directly.
    
    Many popular software solutions for storing documents will also perform textual embedding for you
    (like ChromaDB), however, not all solutions will perform this for you.
"""

"""
    For this lab we will make use of the HuggingFaceEmbedding class from the langchain library. This
    class makes use of "sentence-transformers/all-mpnet-base-v2" as the embedding model by default, so
    there is no need to perform any extra configuration on the class when initializing it.
"""
from langchain.embeddings import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings()

"""
    Once the HuggingFaceEmbeddings class is initialized, we can use it to embed documents and queries.
"""

# embedding multiple documents (or basic strings in this case)
embeddings = embedder.embed_documents(
    [
        "The Trailblazers beat the Lakers in the basketball game last night.",
        "Our game idea is simple: bake a cake without eggs.",
        "Who knew the biggest video game this century would be a pixel side scroller?"
    ],    
)

# embedding a single query
embedded_query = embedder.embed_query("what's the most popular video game?")

"""
    The output of the documents embedding method will return a list of arrays, each of which
    contains multiple Float values that represents the meaning of the text of the document. The query
    method will return a single list that stores Float values representing the query.
"""
print(f"size of embeddings list: {len(embeddings)}, size of each embedding: {len(embeddings[0]), len(embeddings[1]), len(embeddings[2])}")
print(f"size of embedded query: {len(embedded_query)}")

"""
    Take what you learned and implement the methods below to complete the lab.
"""
def embed_documents(documents: list[str]) -> list[float]:
    # TODO: embed the documents passed in the documents list and return the embeddings
    pass

def embed_query(query: str) -> list[float]:
    # TODO: embed the query passed in the query string and return the embedding
    pass

def organize_documents_and_embeddings(documents: list[str]) -> list[list[int, str, list[float]]]:
    """
        TODO:Use the embed_documents method to embed the documents and organize them with their text and 
        an id inside of a list. Return the organized data inside a list.
    """
    pass
        
    