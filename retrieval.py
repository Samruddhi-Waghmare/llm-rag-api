from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

with open("docs.txt", encoding="utf-8") as f:
    text = f.read()

chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
print("Number of chunks:", len(chunks))

chunk_vectors = embedder.encode(chunks, normalize_embeddings=True)
print("Vectors shape:", chunk_vectors.shape)

def retrieve(question, k=2):
    q_vector = embedder.encode([question], normalize_embeddings=True)[0]
    scores = chunk_vectors @ q_vector
    top_indices = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), chunks[i]) for i in top_indices]

if __name__ == "__main__":
    questions = [
        "When can I get my money back?",
        "What time do you close on Friday?",
        "Is there food for people allergic to nuts?",
        "Do you sell pizza?",
    ]

    for q in questions:
        print("\nQUESTION:", q)
        for score, chunk in retrieve(q):
            print(f"  {score:.3f}  {chunk[:70]}...")