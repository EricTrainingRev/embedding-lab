

import unittest
from src.main.lab import embed_documents, embed_query, organize_documents_and_embeddings

class TestTextEmbedding(unittest.TestCase):
    
    test_documents = [
        "The spy was caught red-handed with a crayon in his hand, trying to draw a cake on the wall.",
        "The cake was so delicious that the spy decided to use a crayon to write down the recipe.",
        "The spy was disguised as a baker, but his cover was blown when he accidentally drew a crayon sketch of a spy on the cake.",
        "The cake was decorated with a crayon drawing of a spy, which made the guests feel like they were part of a secret mission.",
        "The spy used a crayon to draw a map of the cake, which helped him locate the hidden microphones and cameras."
    ]
    
    def test_embed_documents(self):
        embeddings = embed_documents(self.test_documents)
        self.assertIsNotNone(embeddings)
        self.assertEqual(len(embeddings), 5)
        for embedding in embeddings:
            self.assertEqual(len(embedding), 768)
            
    def test_embed_query(self):
        query = "what's the most popular video game?"
        query_embedding = embed_query(query)
        self.assertIsNotNone(query_embedding)
        self.assertEqual(len(query_embedding), 768)
    
    def test_organize_documents_and_embeddings(self):
        organized_data = organize_documents_and_embeddings(self.test_documents)
        self.assertIsNotNone(organized_data)
        self.assertEqual(len(organized_data), 5)
        for doc in organized_data:
            self.assertEqual(len(doc), 3)
            self.assertIsInstance(doc[0], int)
            self.assertIsInstance(doc[1], str)
            self.assertIsInstance(doc[2], list)
            self.assertEqual(len(doc[2]), 768)