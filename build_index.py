import os
import glob
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle

DATA_DIR = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
INDEX_FILE = "faiss.index"
META_FILE = "metadata.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBED_MODEL_NAME)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

documents = []
metadata = []

print("Разбиваем тексты на чанки...")

for filepath in tqdm(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    filename = os.path.basename(filepath)
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadata.append({
            "source": filename,
            "chunk_id": i
        })

print(f"Всего чанков: {len(documents)}")

embeddings = model.encode(documents, show_progress_bar=True)

# Создание индекса
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Сохранение индекса и метаданных
faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump(metadata, f)

print("Индекс и метаданные сохранены!")