"""
День 4. Температура
===================
Интерактивный CLI: вводишь запрос — получаешь ответы при пяти значениях
temperature (0, 0.7, 1.2, 1.5, 2.0) с анализом точности, креативности
и отклонения от детерминированного ответа.

Затем AI-судья оценивает все ответы и даёт рекомендации.
"""

import os
import re
import textwrap
import time
from collections import Counter
from math import sqrt
from openai import APIError, OpenAI

# ── Настройка ──────────────────────────────────────────────────────────────────

api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
if not api_key:
    raise RuntimeError(
        "ANTHROPIC_AUTH_TOKEN не найден — проверьте экспорт переменной"
    )

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

SYSTEM_BASE = "You are a helpful assistant. Do not use emojis in your responses."

TEMPERATURES = [0.0, 0.7, 1.2, 1.5, 2.0]

# ── Вызов API ──────────────────────────────────────────────────────────────────


def ask(messages, temperature=0.0, **kwargs):
    """Отправляет запрос к DeepSeek и возвращает текст ответа + finish_reason."""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    choice = resp.choices[0]
    return choice.message.content or "", choice.finish_reason


# ── Эксперимент ────────────────────────────────────────────────────────────────


def make_result(temperature, text, finish, duration):
    """Создаёт словарь результата одного запуска."""
    words = text.split() if text else []
    return {
        "temperature": temperature,
        "text": text.strip(),
        "finish": finish,
        "duration": duration,
        "char_count": len(text),
        "word_count": len(words),
    }


def run_temperature(query, temperature):
    """Один прогон промпта при заданной температуре."""
    t0 = time.time()
    try:
        text, finish = ask(
            messages=[
                {"role": "system", "content": SYSTEM_BASE},
                {"role": "user", "content": query},
            ],
            temperature=temperature,
        )
    except (APIError, ConnectionError, TimeoutError) as e:
        text = f"[ОШИБКА: {e}]"
        finish = "error"
    duration = time.time() - t0
    return make_result(temperature, text, finish, duration)


# ── Метрики ────────────────────────────────────────────────────────────────────


def _tokenize(text):
    """Разбивает текст на слова в нижнем регистре (только буквенно-цифровые)."""
    return re.findall(r"\w+", text.lower())


def calc_type_token_ratio(text):
    """Type-Token Ratio: отношение уникальных слов к общему количеству слов."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _word_freq_vector(text):
    """Строит вектор частот слов (Counter)."""
    return Counter(_tokenize(text))


def cosine_similarity(text_a, text_b):
    """Косинусное сходство между двумя текстами на основе частот слов (0..1)."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vec_a = _word_freq_vector(text_a)
    vec_b = _word_freq_vector(text_b)
    all_words = set(vec_a) | set(vec_b)
    dot = sum(vec_a[w] * vec_b[w] for w in all_words)
    norm_a = sqrt(sum(v * v for v in vec_a.values()))
    norm_b = sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── AI-судья ──────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = textwrap.dedent("""\
    Ты — беспристрастный судья. Твоя задача — оценить, как разные значения
    temperature влияют на ответы AI на один и тот же промпт.

    Тебе будут предоставлены:
    1. Исходный запрос пользователя
    2. 5 ответов — по одному при каждой температуре (0.0, 0.7, 1.2, 1.5, 2.0)

    Оцени каждый ответ по следующим критериям
    (оценка от 1 до 10, можно с одним знаком после запятой):

    1. ACCURACY (Точность) — насколько ответ фактологически верен, релевантен
       и соответствует запросу пользователя?
    2. CREATIVITY (Креативность) — насколько ответ оригинален, образен
       и лексически богат? Обрати внимание на необычные аналогии, метафоры,
       примеры и разнообразие словарного запаса.

    Формат вывода (строго):

    Temperature 0.0:
      Accuracy: X/10  Creativity: Y/10
      Avg: A/10
      Note: <одно предложение>

    Temperature 0.7:
      Accuracy: X/10  Creativity: Y/10
      Avg: A/10
      Note: <одно предложение>

    Temperature 1.2:
      Accuracy: X/10  Creativity: Y/10
      Avg: A/10
      Note: <одно предложение>

    Temperature 1.5:
      Accuracy: X/10  Creativity: Y/10
      Avg: A/10
      Note: <одно предложение>

    Temperature 2.0:
      Accuracy: X/10  Creativity: Y/10
      Avg: A/10
      Note: <одно предложение>

    === Summary ===
    Best temperature for accuracy: <значение>
    Best temperature for creativity: <значение>
    Overall recommendation: <2-3 предложения о том, когда использовать каждую температуру>
""")


def judge_responses(query, all_results):
    """
    Отдельный вызов API (temperature=0) для оценки всех ответов судьёй.
    all_results — список ответов {temperature, text, ...}.
    """
    parts = [f"=== USER QUERY ===\n{query}\n"]
    for r in all_results:
        label = f"Temperature {r['temperature']}"
        parts.append(f"=== {label} ===\n{r['text']}\n")
    judge_input = "\n".join(parts)

    text, finish = ask(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": judge_input},
        ],
        temperature=0.0,
    )
    return {"judge_text": text, "finish": finish}


def parse_judge_scores(judge_text):
    """
    Извлекает оценки и рекомендации из вердикта судьи.
    """
    result = {
        "scores": {},
        "best_accuracy": None,
        "best_creativity": None,
        "recommendation": "",
    }

    lines = judge_text.split("\n")
    current_temp = None

    for line in lines:
        temp_match = re.match(r"Temperature\s+(\d+\.?\d*):", line)
        if temp_match:
            current_temp = float(temp_match.group(1))
            result["scores"][current_temp] = {}
            continue

        if current_temp is None:
            continue

        scores_match = re.search(
            r"Accuracy:\s*([\d.]+).*?Creativity:\s*([\d.]+)",
            line,
        )
        if scores_match:
            result["scores"][current_temp]["accuracy"] = float(scores_match.group(1))
            result["scores"][current_temp]["creativity"] = float(scores_match.group(2))
            continue

        avg_match = re.search(r"Avg:\s*([\d.]+)", line)
        if avg_match and current_temp in result["scores"]:
            result["scores"][current_temp]["avg"] = float(avg_match.group(1))

    _parse_summary_block(lines, result)

    return result


def _parse_summary_block(lines, result):
    """Извлекает итоговые рекомендации из Summary-блока вердикта судьи."""
    summary_lines = []
    in_summary = False
    for line in lines:
        if "=== Summary ===" in line or "===Summary===" in line:
            in_summary = True
            continue
        if in_summary:
            summary_lines.append(line.strip())

    for line in summary_lines:
        best_acc = re.search(r"Best temperature for accuracy:\s*([\d.]+)", line)
        if best_acc:
            result["best_accuracy"] = float(best_acc.group(1))
            continue
        best_cre = re.search(r"Best temperature for creativity:\s*([\d.]+)", line)
        if best_cre:
            result["best_creativity"] = float(best_cre.group(1))
            continue
        rec_match = re.search(r"Overall recommendation:\s*(.+)", line)
        if rec_match:
            result["recommendation"] = rec_match.group(1)

    if not result["recommendation"] and summary_lines:
        result["recommendation"] = " ".join(
            s for s in summary_lines if not re.match(
                r"Best temperature|Overall recommendation:", s
            )
        ).strip()


# ── Вывод ──────────────────────────────────────────────────────────────────────

BOX_WIDTH = 78


def print_separator(char="="):
    """Рисует горизонтальную линию-разделитель."""
    print(char * (BOX_WIDTH + 2))


def print_header(title):
    """Заголовок секции с обрамлением."""
    print()
    print_separator("=")
    print(f"  {title}")
    print_separator("=")
    print()


def print_progress(label):
    """Сообщение о ходе выполнения."""
    print(f"  >>> {label}...")


def print_format_label(temp):
    """Форматирует заголовок для заданной температуры."""
    return f"T={temp}"


def print_all_responses(all_results):
    """Выводит все ответы подряд с меткой температуры."""
    print_header("Все ответы")
    for r in all_results:
        label = print_format_label(r["temperature"])
        print(f"┌─ {label} ({r['word_count']} слов, {r['duration']:.1f} с)")
        for line in r["text"].split("\n"):
            print(f"│ {line}")
        print(f"└{'─' * BOX_WIDTH}")
        print()


def print_metrics_table(all_results, parsed):
    """Сводная таблица сравнения всех температур."""
    print_header("Сводка метрик")

    # --- Объективные метрики ---
    print("  Объективные метрики:")
    print()

    baseline = None
    for r in all_results:
        if r["temperature"] == 0.0:
            baseline = r["text"]
            break

    col_width = 10
    _print_table_header(col_width)

    for name, getter in [
        ("Длина (слов)", lambda r: f"{r['word_count']}"),
        ("TTR (лексическое богатство)",
         lambda r: f"{calc_type_token_ratio(r['text']):.3f}"),
        ("Время (с)", lambda r: f"{r['duration']:.1f}"),
    ]:
        vals = [getter(r).rjust(col_width) for r in all_results]
        print(f"  {name:<30}", *vals)

    if baseline:
        sim_vals = []
        for r in all_results:
            sim = cosine_similarity(r["text"], baseline)
            sim_vals.append(f"{sim:.3f}".rjust(col_width))
        print(f"  {'Сходство с T=0':<30}", *sim_vals)

    print()

    # --- AI-оценки ---
    if parsed["scores"]:
        _print_ai_scores(parsed, col_width)


def _print_table_header(col_width):
    """Выводит заголовок таблицы."""
    h_parts = ["  Параметр".ljust(30)]
    for t in TEMPERATURES:
        h_parts.append(f"T={t}".rjust(col_width))
    print("".join(h_parts))
    print("  " + "─" * 30 + " " + " ".join("─" * col_width for _ in TEMPERATURES))


def _print_ai_scores(parsed, col_width):
    """Выводит блок оценок AI-судьи."""
    print("  Оценки AI-судьи (из 10):")
    print()
    h_parts = ["  Критерий".ljust(30)]
    for t in TEMPERATURES:
        h_parts.append(f"T={t}".rjust(col_width))
    print("".join(h_parts))
    print("  " + "─" * 30 + " " + " ".join("─" * col_width for _ in TEMPERATURES))

    for criterion in ("accuracy", "creativity", "avg"):
        vals = []
        for t in TEMPERATURES:
            if t in parsed["scores"] and criterion in parsed["scores"][t]:
                s = parsed["scores"][t].get(criterion)
                if s is not None:
                    vals.append(f"{s:.1f}".rjust(col_width))
                else:
                    vals.append("?".rjust(col_width))
            else:
                vals.append("?".rjust(col_width))
        label = {"accuracy": "Точность",
                 "creativity": "Креативность",
                 "avg": "Среднее"}.get(criterion, criterion)
        print(f"  {label:<30}", *vals)
    print()


def print_recommendations(parsed):
    """Выводит рекомендации по использованию temperature."""
    print_header("Рекомендации")

    recs = [
        (0.0, "Точность и воспроизводимость",
         ["Фактологические ответы, перевод, извлечение данных",
          "Генерация кода, математические расчёты",
          "Когда нужен детерминированный, предсказуемый результат"]),
        (0.7, "Баланс точности и креативности",
         ["Чат-боты общего назначения, ассистенты",
          "Написание писем, документов, текстов",
          "Объяснения и обучающие материалы"]),
        (1.2, "Повышенная креативность",
         ["Творческие тексты, сторителлинг",
          "Маркетинговые и рекламные тексты",
          "Генерация нескольких вариантов ответа"]),
        (1.5, "Экспериментальные ответы",
         ["Мозговой штурм, генерация нестандартных идей",
          "Задачи, где необычные формулировки — плюс",
          "Поиск альтернативных решений"]),
        (2.0, "Максимальная свобода (риск галлюцинаций)",
         ["Художественное творчество, поэзия",
          "Когда точность не важна, нужна только оригинальность",
          "С осторожностью: возможны бессвязные ответы"]),
    ]

    for temp, title, bullets in recs:
        suffix = ""
        if parsed.get("best_accuracy") == temp:
            suffix += " [Лучшая точность]"
        if parsed.get("best_creativity") == temp:
            suffix += " [Лучшая креативность]"
        print(f"  Temperature = {temp} — {title}{suffix}")
        for b in bullets:
            print(f"    - {b}")
        print()

    if parsed.get("recommendation"):
        print("  Комментарий AI-судьи:")
        for line in parsed["recommendation"].split(". "):
            line = line.strip()
            if line:
                end = "." if not line.endswith(".") else ""
                print(f"    - {line}{end}")
        print()


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    """Интерактивный CLI: ввод запроса → эксперимент с температурами."""
    print()
    print("=" * (BOX_WIDTH + 2))
    print("  День 4: Температура — сравнение 0 / 0.7 / 1.2 / 1.5 / 2.0")
    print("=" * (BOX_WIDTH + 2))
    print()
    print("  Введи 'exit' или 'quit' для выхода.")
    print()

    while True:
        try:
            query = input("Ваш запрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("  Выход.")
            break
        if query.lower() in ("exit", "quit", "выход"):
            print("  Выход.")
            break
        if not query:
            continue

        _run_experiment(query)


def _run_experiment(query):
    """Полный цикл: эксперименты → судья → вывод."""
    total_calls = len(TEMPERATURES) + 1
    print()
    print_progress(f"Всего {total_calls} вызовов DeepSeek API")
    print()

    # Фаза 1: эксперименты
    all_results = []
    for temp in TEMPERATURES:
        r = run_temperature(query, temp)
        all_results.append(r)
        status = "OK" if r["finish"] != "error" else "ERROR"
        print(f"  [{r['temperature']:.1f}] {r['word_count']} слов, "
              f"{r['duration']:.1f} с ({status})")
    print()

    # Фаза 2: AI-судья
    print_progress("AI-судья оценивает ответы")
    try:
        judge_result = judge_responses(query, all_results)
        parsed = parse_judge_scores(judge_result["judge_text"])
        print("       Готово")
    except (APIError, IndexError, AttributeError) as e:
        print(f"       ОШИБКА: {e}")
        judge_result = {"judge_text": f"Оценка не удалась: {e}",
                        "finish": "error"}
        parsed = {"scores": {}, "best_accuracy": None,
                  "best_creativity": None,
                  "recommendation": ""}
    print()

    # Фаза 3: показ ответов
    print_progress("Показать все ответы? (Enter — да, n — пропустить)")
    try:
        show_all = input("  >>> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        show_all = "n"
    if show_all != "n":
        print_all_responses(all_results)

    # Фаза 4: метрики
    print_metrics_table(all_results, parsed)

    # Фаза 5: вердикт судьи
    print_header("Вердикт AI-судьи")
    for line in judge_result["judge_text"].strip().split("\n"):
        print(f"  {line}")
    print()

    # Фаза 6: рекомендации
    print_recommendations(parsed)

    print_separator("=")
    print()
    print("  Можно ввести новый запрос или exit для выхода.")
    print()


if __name__ == "__main__":
    main()
