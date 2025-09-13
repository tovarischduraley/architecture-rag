import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Загрузка
index = faiss.read_index("faiss.index")
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
model = SentenceTransformer("all-MiniLM-L6-v2")

query = input("🔍 Введите запрос: ")
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=3)

print("\n🎯 Найденные чанки:\n")
for idx in I[0]:
    meta = metadata[idx]
    print(f"Файл: {meta['source']} | Чанк: {meta['chunk_id']}")