# День 5: Версии моделей (Weak / Medium / Strong)

## Задание

Выполните один и тот же запрос на трёх моделях разной мощности через OpenRouter API.

Модели:
| Уровень | Модель | Провайдер | Цена (вход/выход за 1M токенов) |
|---------|--------|-----------|:-------------------------------:|
| Слабая (Weak) | Gemma 3 4B | Google | $0.04 / $0.08 |
| Средняя (Medium) | Llama 3.1 8B | Meta | $0.02 / $0.03 |
| Сильная (Strong) | Gemini 2.5 Flash Lite | Google | $0.10 / $0.40 |

### Замеры
- время ответа (сек)
- количество токенов (входных и выходных)
- скорость генерации (токенов/c)
- стоимость ($)

### Сравнение
- качество ответов
- скорость API
- цена

## Подготовка

### 1. Получить API ключ OpenRouter

Зарегистрируйтесь на [openrouter.ai](https://openrouter.ai), создайте API ключ в настройках.

### 2. Сохранить ключ

**Вариант А — macOS Keychain (рекомендуется):**
```bash
security add-generic-password -s 'openrouter-api-key' -w 'sk-or-v1-...'
```

**Вариант Б — переменная окружения:**
```bash
export OPENROUTER_API_KEY='sk-or-v1-...'
```

### 3. Активировать окружение

```bash
source .venv/bin/activate
```

## Использование

Скрипт работает в **интерактивном режиме**:

```bash
python3 week-1/day-5/model_comparison.py
```

Вводишь запрос — получаешь ответы от всех трёх моделей + таблицу метрик.
Для выхода: `exit` / `quit` / `выход`.

### Режим одного запроса (без интерактива)

```bash
python3 week-1/day-5/model_comparison.py --prompt "Объясни ML за 30 секунд"
```

Результаты автоматически сохраняются в `week-1/day-5/output/NNN-тема/`:
- `results.json` — сырые данные (токены, время, стоимость)
- `results.md` — сравнительная таблица и ссылки на модели

Для сохранения в произвольный файл используйте `--output`:

```bash
python3 week-1/day-5/model_comparison.py --prompt "Что такое тензор?" --output custom.json
```

## Параметры

| Параметр | Описание | По умолчанию |
|---|---|---|
| `--prompt, -p` | Одиночный запрос (без интерактива) | (интерактив) |
| `--max-tokens` | Макс. токенов на ответ | 256 |
| `--temperature, -t` | Температура семплинга | 0.7 |
| `--output, -o` | Сохранить в указанный файл (по умолч. авто-сохранение в `output/`) | авто `output/ID-тема/` |

## Ссылки

- [Gemma 3 4B](https://openrouter.ai/models/google/gemma-3-4b-it)
- [Llama 3.1 8B](https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct)
- [Gemini 2.5 Flash Lite](https://openrouter.ai/models/google/gemini-2.5-flash-lite)
- [OpenRouter](https://openrouter.ai)
