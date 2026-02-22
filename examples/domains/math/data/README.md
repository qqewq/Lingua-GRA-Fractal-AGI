# Math domain data / Данные для математического домена

EN  
This folder is intended for a **small math corpus** used in Lingua‑GRA‑Math examples.

Recommended structure:

- `raw/` – raw texts (definitions, theorems, proofs, notes).  
- `processed/` – tokenized / cleaned versions, ready for embedding.  

You can start with:

- a few pages of lecture notes or textbooks (plain text),  
- or a synthetic corpus of short mathematical statements and explanations.

The example script `train_lingua_gra_math.py` assumes that a helper
like `load_math_corpus_embeddings(data_path, ...)` can load or build
embeddings from files in this directory.

---

RU  
Эта папка предназначена для **маленького математического корпуса**, который используется в примерах Lingua‑GRA‑Math.

Рекомендуемая структура:

- `raw/` – сырые тексты (определения, теоремы, доказательства, конспекты).  
- `processed/` – преобразованные / токенизированные тексты, готовые к получению эмбеддингов.  

Можно начать с:

- нескольких страниц конспектов или учебника (обычный текст),  
- или синтетического корпуса коротких математических утверждений и объяснений.

Скрипт `train_lingua_gra_math.py` предполагает, что вспомогательная функция
вроде `load_math_corpus_embeddings(data_path, ...)` сможет загрузить или построить
эмбеддинги на основе файлов из этой директории.[web:416]
