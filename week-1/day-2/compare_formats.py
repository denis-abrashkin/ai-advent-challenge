"""
День 2. Формат ответа
=====================
Интерактивный CLI: вводишь запрос — получаешь четыре варианта ответа
с разным уровнем контроля:
  1. Без ограничений
  2. API-контроль   (max_tokens + stop через API-параметры)
  3. Prompt-контроль (формат, лимит слов, стоп-маркер в промпте)
  4. Гибрид          (max_tokens + stop через API, формат через промпт)
"""

import os
import textwrap
from openai import OpenAI

# ── Настройка ──────────────────────────────────────────────────────────────

api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
if not api_key:
    raise RuntimeError(
        "ANTHROPIC_AUTH_TOKEN не найден — проверьте экспорт переменной"
    )

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

STOP_MARKER = "[DONE]"


# ── Вызов API ──────────────────────────────────────────────────────────────


def ask(messages, **kwargs):
    """Отправляет запрос к DeepSeek и возвращает текст ответа + finish_reason."""
    resp = client.chat.completions.create(
        model="deepseek-chat", messages=messages, **kwargs
    )
    choice = resp.choices[0]
    return choice.message.content or "", choice.finish_reason


# ── Четыре варианта ────────────────────────────────────────────────────────


def run_baseline(query):
    """Без ограничений."""
    text, finish = ask(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
    )
    return {
        "title": "1. Без ограничений",
        "params": "—",
        "text": text,
        "finish": finish,
    }


def run_api_control(query):
    """Ограничения через API: max_tokens + stop, формат в промпте."""
    text, finish = ask(
        messages=[
            {
                "role": "system",
                "content": textwrap.dedent(f"""\
                    Ты — ассистент. Отвечай строго в следующем формате:

                    Заголовок: <один заголовок>
                    Ключевые тезисы:
                    - <тезис 1>
                    - <тезис 2>
                    - <тезис 3>
                    Вывод: <одно предложение>

                    В конце напиши '{STOP_MARKER}'.
                """),
            },
            {"role": "user", "content": query},
        ],
        max_tokens=250,
        stop=[STOP_MARKER],
    )
    return {
        "title": "2. API-контроль",
        "params": "max_tokens=250, stop=['[DONE]']",
        "text": text,
        "finish": finish,
    }


def run_prompt_control(query):
    """Ограничения через промпт: формат, лимит слов, стоп-маркер."""
    text, finish = ask(
        messages=[
            {
                "role": "system",
                "content": textwrap.dedent(f"""\
                    Ты — ассистент. Отвечай строго в следующем формате:

                    Заголовок: <один заголовок>
                    Ключевые тезисы:
                    - <тезис 1>
                    - <тезис 2>
                    - <тезис 3>
                    Вывод: <одно предложение>

                    Ограничения:
                    - Ответ не длиннее 80 слов.
                    - В конце напиши '{STOP_MARKER}' и остановись.
                    - Без эмодзи.
                """),
            },
            {"role": "user", "content": query},
        ],
    )
    return {
        "title": "3. Prompt-контроль",
        "params": "формат + лимит слов + стоп-маркер в промпте",
        "text": text,
        "finish": finish,
    }


def run_hybrid(query):
    """Гибрид: max_tokens+stop через API, формат через промпт."""
    text, finish = ask(
        messages=[
            {
                "role": "system",
                "content": textwrap.dedent(f"""\
                    Ты — ассистент. Отвечай строго в следующем формате:

                    Заголовок: <один заголовок>
                    Ключевые тезисы:
                    - <тезис 1>
                    - <тезис 2>
                    - <тезис 3>
                    Вывод: <одно предложение>

                    В конце напиши '{STOP_MARKER}'.
                """),
            },
            {"role": "user", "content": query},
        ],
        max_tokens=250,
        stop=[STOP_MARKER],
    )
    return {
        "title": "4. Гибрид (API + Prompt)",
        "params": "max_tokens=250, stop=['[DONE]'] + формат в промпте",
        "text": text,
        "finish": finish,
    }


# ── Вывод ──────────────────────────────────────────────────────────────────


def print_result(result):
    """Выводит один вариант: заголовок, параметры, ответ, метрики."""
    print(f"┌─ {result['title']}")
    print(f"│ Параметры: {result['params']}")
    print(f"│ Finish: {result['finish']}")
    print(f"│ Символов: {len(result['text'])} | Слов: {len(result['text'].split())}")
    print("│")
    for line in result["text"].split("\n"):
        print(f"│ {line}")
    print("└" + "─" * 72)
    print()


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    """Интерактивный CLI: ввод запроса → четыре варианта ответа."""
    print()
    print("=" * 74)
    print("  День 2: Четыре уровня контроля над ответом")
    print("=" * 74)
    print()
    print("  Введи 'exit' или 'quit' для выхода.")
    print()

    while True:
        query = input("Запрос: ").strip()
        if query.lower() in ("exit", "quit", "выход"):
            print("  Выход.")
            break
        if not query:
            continue

        print()
        variants = [
            run_baseline(query),
            run_api_control(query),
            run_prompt_control(query),
            run_hybrid(query),
        ]

        for result in variants:
            print_result(result)


if __name__ == "__main__":
    main()
