###  RAG по серии игр DMC

- Модель эмбеддингов: `all-MiniLM-L6-v2`  
  - Размер эмбеддинга: 384
  - [Модель в Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Использовано файлов: 28 (все страницы из dmc wiki)
- Кол-во чанков: ~40
- Время индексации: 20 секунд
- Содержимое:
  - `replacer.py` — скрипт для замены ключевых слов в базе знаний
  - `build_index.py` — скрипт для генерации индекса
  - `faiss.index` — FAISS индекс
  - `metadata.pkl` — метаданные
  - `assistant.py` — RAG бот

