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
- стоимость (расчётная, по ценам HuggingFace Inference API)

### Сравнение
- качество ответов
- скорость инференса
- ресурсоёмкость

## Результаты (M2 Pro, MPS, fp16)

| Метрика | 0.5B | 1.5B | 3B |
|---------|:----:|:----:|:--:|
| Среднее время | 1.2 с | 3.1 с | 7.0 с |
| Средняя скорость | 45.1 ток/с | 31.5 ток/с | 13.1 ток/с |
| Средняя длина ответа | 50 токенов | 95 токенов | 79 токенов |
| Загрузка модели | 2.9 с | 6.2 с | ~10 с |

**Выводы:**
- Скорость падает с размером модели: 0.5B в **3.4× быстрее** 3B
- 1.5B — золотая середина: хорошее качество за 3.1 с
- 3B даёт наиболее точные и структурированные ответы

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
и кешируются в `~/.cache/huggingface/hub/`.

> **ℹ️ Для Apple Silicon (MPS):** `torch` автоматически использует Metal Performance Shaders.
  Для ускорения загрузки моделей рекомендуется установить `HF_TOKEN`:
  ```bash
  export HF_TOKEN="ваш_токен"
  ```
  Получить токен: https://huggingface.co/settings/tokens

## Использование

```bash
# Все модели на всех тестовых промптах
python3 week-1/day-5/model_comparison.py

# Только слабая модель на одном промпте
python3 week-1/day-5/model_comparison.py --model weak --prompt "Расскажи о ML"

# Сохранение результатов
python3 week-1/day-5/model_comparison.py --output results.json
```

## Параметры

| Параметр | Описание | По умолчанию |
|---|---|---|
| `--prompt, -p` | Один кастомный промпт | (все тестовые) |
| `--model, -m` | Фильтр моделей (weak, medium, strong) | (все три) |
| `--device, -d` | Устройство (cpu, mps, cuda) | автоопределение |
| `--max-tokens` | Макс. токенов на ответ | 256 |
| `--temperature, -t` | Температура семплинга | 0.7 |
| `--output, -o` | Путь к JSON для сохранения | (нет) |

## Тестовые промпты

1. "Explain the difference between supervised and unsupervised learning in two sentences."
2. "Write a short poem about artificial intelligence (4 lines max)."
3. "What is 124 * 37? Show your reasoning step by step."

## Ссылки

- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
