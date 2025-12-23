## Локальный индекс документов с эмбеддингами

Что это делает:

- Берёт файлы (README/статьи/код/`pdf`→текст)
- Режет текст на чанки (с overlap)
- Строит эмбеддинги локальной моделью `sentence-transformers`
- Сохраняет индекс на диск (векторный массив + метаданные)

### Установка

Из корня репозитория:

```bash
source new-env/bin/activate
python3 -m pip install -r requirements_rag.txt
```

### Сборка индекса

Пример: проиндексировать README и код Kotlin:

```bash
python3 tools/build_doc_index.py \
  --input README.md \
  --input telegram_chat_bot/README.md \
  --input-dir mcp-server-kotlin/src \
  --ext kt --ext md \
  --store dir \
  --out doc_index
```

Варианты хранения индекса:

- `--store dir` → папка `doc_index/`
- `--store json` → один файл `index.json`
- `--store sqlite` → один файл `index.sqlite`

#### Формат `dir`

Появится папка `doc_index/`:

- `meta.json` — параметры (модель, размер чанка, размерность)
- `chunks.jsonl` — текстовые чанки + источник
- `embeddings.npy` — матрица эмбеддингов `N x D` (float32, нормализована)

#### Формат `json` (один файл)

```bash
python3 tools/build_doc_index.py \
  --input telegram_chat_bot/README.md \
  --store json \
  --out doc_index/index.json
```

#### Формат `sqlite` (один файл)

```bash
python3 tools/build_doc_index.py \
  --input telegram_chat_bot/README.md \
  --store sqlite \
  --out doc_index/index.sqlite
```

### Быстрая проверка (поиск)

```bash
python3 tools/search_doc_index.py --index doc_index --query "как работает mcp сервер" --top-k 5
```

### Один текстовый файл как “база знаний” для агента

В репозитории уже создан пустой файл:

- `kb/knowledge_base.txt` — заполни его своим текстом.

После этого собери индекс (например, в SQLite):

```bash
python3 tools/build_doc_index.py \
  --input kb/knowledge_base.txt \
  --store sqlite \
  --out doc_index/knowledge_base.sqlite
```

Дальше можно “спрашивать по базе” (сначала он покажет найденный контекст; если настроен YandexGPT — ещё и ответит):

```bash
python3 tools/rag_ask.py \
  --index doc_index/knowledge_base.sqlite \
  --question "Что написано про ...?" \
  --top-k 5
```

### Агент: сравнение ответа модели с RAG и без RAG

Команда ниже запускает “агента” с двумя режимами и сравнением:

```bash
python3 rag_agent.py \
  --index doc_index/knowledge_base.sqlite \
  --mode compare \
  --question "Что изменилось в оплате продуктов с 1 марта 2024?"
```

Что он делает:

- вопрос → поиск релевантных чанков (top-k)
- сбор контекста
- запрос к LLM **без RAG** (только вопрос)
- запрос к LLM **с RAG** (вопрос + найденный контекст)
- вывод сравнения + (опционально) автоматический вывод “где RAG помог/где нет” через LLM-judge

Переменные окружения для запросов к YandexGPT:

- `YANDEX_FOLDER_ID`
- `YANDEX_AUTH`

### Примечания

- По умолчанию используется мультиязычная модель: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Для PDF нужен `pypdf` (он уже в `requirements_rag.txt`)


