# Git MCP Server

MCP (Model Context Protocol) сервер для интеграции с Git-репозиторием Bookechi.

## Описание

Сервер предоставляет инструменты для:
- Получения информации о текущей ветке и статусе репозитория
- Просмотра истории коммитов
- Чтения файлов из репозитория
- Поиска по коду
- Просмотра изменений (diff)

## Инструменты (Tools)

### `git_branch`
Получить имя текущей ветки.

### `git_status`
Получить статус репозитория — staged, modified и untracked файлы.

### `git_log`
Получить последние коммиты.
- `count` (int, optional): количество коммитов (по умолчанию 10, максимум 50)

### `git_diff`
Получить diff для файла или коммита.
- `file_path` (string, optional): путь к файлу
- `commit` (string, optional): хеш коммита

### `read_file`
Прочитать содержимое файла.
- `path` (string, required): путь к файлу относительно корня репозитория

### `list_files`
Получить список файлов в директории.
- `directory` (string, optional): путь к директории (по умолчанию корень)
- `extension` (string, optional): фильтр по расширению (например, ".kt")
- `max_depth` (int, optional): максимальная глубина (по умолчанию 3)

### `search_code`
Поиск по коду.
- `pattern` (string, required): паттерн поиска (grep синтаксис)
- `file_extension` (string, optional): фильтр по расширению
- `max_results` (int, optional): максимум результатов (по умолчанию 20)

### `file_history`
Получить историю коммитов для файла.
- `path` (string, required): путь к файлу
- `count` (int, optional): количество коммитов (по умолчанию 5)

## Запуск

### Как stdio MCP сервер
```bash
python3 git_mcp_server.py
```

### С указанием репозитория
```bash
GIT_REPO_PATH=/path/to/repo python3 git_mcp_server.py
```

## Протокол

Сервер использует JSON-RPC 2.0 через stdio (stdin/stdout) согласно спецификации MCP.

### Пример запроса
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
```

### Пример вызова инструмента
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "read_file",
    "arguments": {
      "path": "app/src/main/java/fuwafuwa/time/bookechi/data/model/Book.kt"
    }
  }
}
```

## Интеграция с Telegram-ботом

В боте доступны команды для работы с Git:
- `/git_status` — статус репозитория
- `/git_branch` — текущая ветка
- `/git_log` — последние коммиты
- `/git_files [путь]` — файлы в директории
- `/git_show <файл>` — показать содержимое файла

Команда `/help` использует RAG + Git для ответов на вопросы о проекте.



