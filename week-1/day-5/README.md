# День 5: Версии моделей (Weak / Medium / Strong)

## Задание

Выполните один и тот же запрос на трёх моделях разной мощности из HuggingFace.

Модели (семейство Qwen2.5 для единообразия архитектуры):
| Уровень | Модель | Параметры | Память (fp16) |
|---------|--------|:---------:|:-------------:|
| Слабая (Weak) | Qwen2.5-0.5B-Instruct | 0.5B | ~350 MB |
| Средняя (Medium) | Qwen2.5-1.5B-Instruct | 1.5B | ~1.0 GB |
| Сильная (Strong) | Qwen2.5-3B-Instruct | 3.0B | ~2.0 GB |

### Замеры
- время ответа (сек)
- количество токенов (входных и выходных)
- скорость генерации (токенов/c)

### Сравнение
- качество ответов
- скорость инференса
- ресурсоёмкость (параметры, память)

## Установка

Проект использует виртуальное окружение Python. Зависимости:

```bash
# Активировать виртуальное окружение
source .venv/bin/activate

# Установить зависимости (из корня проекта)
pip install -r requirements.txt
pip install transformers torch accelerate sentencepiece
```

Модели скачиваются из HuggingFace Hub автоматически при первом запуске
и кешируются в `dev/.huggingface/hub/`.

> **ℹ️ Для Apple Silicon (MPS):** `torch` автоматически использует Metal Performance Shaders.
  Для ускорения загрузки моделей можно установить `HF_TOKEN`:
  ```bash
  export HF_TOKEN="ваш_токен"
  ```
  Получить токен: https://huggingface.co/settings/tokens

> **ℹ️** `__pycache__` отключён через `PYTHONDONTWRITEBYTECODE=1` в `.venv/bin/activate`.

## Использование

Скрипт работает в **интерактивном режиме** (как day-1 — day-4):

```bash
# Активировать окружение и запустить
source .venv/bin/activate
python3 week-1/day-5/model_comparison.py
```

Вводишь запрос — получаешь ответы от всех трёх моделей + таблицу метрик.
Для выхода: `exit` / `quit` / `выход`.

### Режим одного запроса (без интерактива)

```bash
python3 week-1/day-5/model_comparison.py --prompt "Объясни ML за 30 секунд"
```

Результаты автоматически сохраняются в `week-1/day-5/output/NNN-тема/`:
- `results.json` — сырые данные
- `results.md` — сравнительная таблица и ссылки на модели

Для сохранения в произвольный файл используйте `--output`:

```bash
python3 week-1/day-5/model_comparison.py --prompt "Что такое тензор?" --output custom.json
```

## Параметры

| Параметр | Описание | По умолчанию |
|---|---|---|
| `--prompt, -p` | Одиночный запрос (без интерактива) | (интерактив) |
| `--device, -d` | Устройство: cpu, mps, cuda | автоопределение |
| `--max-tokens` | Макс. токенов на ответ | 256 |
| `--temperature, -t` | Температура семплинга | 0.7 |
| `--output, -o` | Сохранить в указанный файл (по умолч. авто-сохранение в `output/`) | авто `output/ID-тема/` |

## Ссылки

- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
