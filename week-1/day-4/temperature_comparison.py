"""
День 4. Температура
===================
Интерактивный CLI: вводишь запрос — получаешь ответы при трёх значениях
temperature (0, 0.7, 1.2) с анализом точности, креативности и разнообразия.

Каждый температурный режим запускается 3 раза для измерения diversity.
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

TEMPERATURES = [0.0, 0.7, 1.2]
RUNS_PER_TEMP = 3

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


def make_run(temperature, run_num, text, finish, duration):
    """Создаёт словарь результата одного запуска."""
    words = text.split() if text else []
    return {
        "temperature": temperature,
        "run": run_num,
        "text": text.strip(),
        "finish": finish,
        "duration": duration,
        "char_count": len(text),
        "word_count": len(words),
    }


def run_temperature(query, temperature, runs=RUNS_PER_TEMP):
    """Запускает промпт при заданной температуре N раз. Возвращает список RunResult."""
    results = []
    for i in range(runs):
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
        results.append(make_run(temperature, i + 1, text, finish, duration))
    return results


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


def calc_consistency(group):
    """
    Среднее косинусное сходство между всеми парами в группе (0..1).
    1 = все ответы идентичны, 0 = полностью разные.
    """
    if len(group) < 2:
        return 1.0
    sims = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            sims.append(cosine_similarity(group[i]["text"], group[j]["text"]))
    return sum(sims) / len(sims) if sims else 1.0


def calc_diversity(group):
    """Разнообразие = 1 - consistency."""
    return 1.0 - calc_consistency(group)


# ── AI-судья ──────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = textwrap.dedent("""\
    You are an impartial judge evaluating how different temperature settings
    affect an AI's responses to the same prompt.

    You will be given:
    1. The original user query
    2. 9 responses — 3 runs at each temperature (0.0, 0.7, 1.2)

    Evaluate each group of 3 responses per temperature on these criteria
    (score each from 1 to 10, use 1 decimal place if needed):

    1. ACCURACY — how factually correct, relevant, and on-topic are the
       responses given the user's query?
    2. CREATIVITY — how original, imaginative, and linguistically rich are
       the responses? Look for unique analogies, metaphors, examples, and
       vocabulary variety.
    3. CONSISTENCY — how similar are the 3 responses within this temperature
       group? (High consistency = nearly identical answers; low = diverse.)

    Output format (strict):

    Temperature 0.0:
      Accuracy: X/10  Creativity: Y/10  Consistency: Z/10
      Avg: A/10
      Note: <one sentence>

    Temperature 0.7:
      Accuracy: X/10  Creativity: Y/10  Consistency: Z/10
      Avg: A/10
      Note: <one sentence>

    Temperature 1.2:
      Accuracy: X/10  Creativity: Y/10  Consistency: Z/10
      Avg: A/10
      Note: <one sentence>

    === Summary ===
    Best temperature for accuracy: <value>
    Best temperature for creativity: <value>
    Best temperature for consistency: <value>
    Overall recommendation: <2-3 sentences about when to use each temperature>
""")


def judge_responses(query, all_results):
    """
    Отдельный вызов API (temperature=0) для оценки всех ответов судьёй.
    all_results — плоский список ответов со всех температур.
    """
    parts = [f"=== USER QUERY ===\n{query}\n"]
    for r in all_results:
        label = f"Temperature {r['temperature']}, Run {r['run']}"
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
    Возвращает словарь с баллами по каждой температуре и рекомендациям.
    """
    result = {
        "scores": {},
        "best_accuracy": None,
        "best_creativity": None,
        "best_consistency": None,
        "recommendation": "",
    }

    lines = judge_text.split("\n")
    current_temp = None

    for line in lines:
        # Определяем текущую температуру
        temp_match = re.match(r"Temperature\s+(\d+\.?\d*):", line)
        if temp_match:
            current_temp = float(temp_match.group(1))
            result["scores"][current_temp] = {}
            continue

        if current_temp is None:
            continue

        # Accuracy / Creativity / Consistency
        scores_match = re.search(
            r"Accuracy:\s*([\d.]+).*?Creativity:\s*([\d.]+).*?Consistency:\s*([\d.]+)",
            line,
        )
        if scores_match:
            result["scores"][current_temp]["accuracy"] = float(scores_match.group(1))
            result["scores"][current_temp]["creativity"] = float(scores_match.group(2))
            result["scores"][current_temp]["consistency"] = float(scores_match.group(3))
            continue

        # Avg
        avg_match = re.search(r"Avg:\s*([\d.]+)", line)
        if avg_match and current_temp in result["scores"]:
            result["scores"][current_temp]["avg"] = float(avg_match.group(1))

    # Ищем Summary-блок
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
        best_con = re.search(r"Best temperature for consistency:\s*([\d.]+)", line)
        if best_con:
            result["best_consistency"] = float(best_con.group(1))
            continue
        rec_match = re.search(r"Overall recommendation:\s*(.+)", line)
        if rec_match:
            result["recommendation"] = rec_match.group(1)

    if not result["recommendation"] and summary_lines:
        result["recommendation"] = " ".join(
            l for l in summary_lines if not re.match(
                r"Best temperature|Overall recommendation:", l
            )
        ).strip()

    return result


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


def print_results_by_temp(by_temp):
    """Выводит все ответы, сгруппированные по температуре."""
    for temp in TEMPERATURES:
        print_header(f"Temperature = {temp}")
        for r in by_temp[temp]:
            tag = f"Запуск {r['run']} — {r['word_count']} слов, {r['duration']:.1f} с"
            print(f"┌─ {tag}")
            for line in r["text"].split("\n"):
                print(f"│ {line}")
            print(f"└{'─' * BOX_WIDTH}")
            print()


def print_metrics_table(by_temp, parsed):
    """Сводная таблица сравнения всех температур."""
    print_header("Сводка метрик")

    # --- Объективные метрики ---
    print("  Объективные метрики:")
    print()
    header = f"  {'Параметр':<30} {'T=0.0':>10} {'T=0.7':>10} {'T=1.2':>10}"
    print(header)
    print(f"  {'─' * 30} {'─' * 10} {'─' * 10} {'─' * 10}")

    metrics = [
        ("Средняя длина (слов)",
         lambda g: f"{sum(r['word_count'] for r in g) / len(g):.0f}"),
        ("TTR (лексическое богатство)",
         lambda g: f"{sum(calc_type_token_ratio(r['text']) for r in g) / len(g):.3f}"),
        ("Consistency",
         lambda g: f"{calc_consistency(g):.3f}"),
        ("Diversity",
         lambda g: f"{1.0 - calc_consistency(g):.3f}"),
        ("Среднее время (с)",
         lambda g: f"{sum(r['duration'] for r in g) / len(g):.1f}"),
    ]

    for name, getter in metrics:
        vals = [getter(by_temp[t]) for t in TEMPERATURES]
        print(f"  {name:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")
    print()

    # --- AI-оценки ---
    if parsed["scores"]:
        print("  Оценки AI-судьи (из 10):")
        print()
        h = f"  {'Критерий':<30} {'T=0.0':>10} {'T=0.7':>10} {'T=1.2':>10}"
        print(h)
        print(f"  {'─' * 30} {'─' * 10} {'─' * 10} {'─' * 10}")

        for criterion in ("accuracy", "creativity", "consistency", "avg"):
            vals = []
            for t in TEMPERATURES:
                if t in parsed["scores"] and criterion in parsed["scores"][t]:
                    s = parsed["scores"][t].get(criterion)
                    vals.append(f"{s:.1f}" if s is not None else "?")
                else:
                    vals.append("?")
            label = {"accuracy": "Точность",
                     "creativity": "Креативность",
                     "consistency": "Согласованность",
                     "avg": "Среднее"}.get(criterion, criterion)
            print(f"  {label:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")
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
        (1.2, "Максимальная креативность и разнообразие",
         ["Мозговой штурм, генерация идей",
          "Креативное письмо, сторителлинг",
          "Когда нужно много вариантов одного ответа"]),
    ]

    for temp, title, bullets in recs:
        suffix = ""
        if parsed.get("best_accuracy") == temp:
            suffix += " [Лучшая точность]"
        if parsed.get("best_creativity") == temp:
            suffix += " [Лучшая креативность]"
        if parsed.get("best_consistency") == temp:
            suffix += " [Лучшая согласованность]"
        print(f"  Temperature = {temp} — {title}{suffix}")
        for b in bullets:
            print(f"    - {b}")
        print()

    if parsed.get("recommendation"):
        print(f"  Комментарий AI-судьи:")
        for line in parsed["recommendation"].split(". "):
            line = line.strip()
            if line:
                print(f"    - {line}." if not line.endswith(".") else f"    - {line}")
        print()


def print_all_responses_grouped(all_results):
    """Выводит все ответы подряд с меткой температуры для быстрого просмотра."""
    print_header("Все ответы")
    for r in all_results:
        label = f"T={r['temperature']}, Run {r['run']}"
        print(f"┌─ {label} ({r['word_count']} слов)")
        for line in r["text"].split("\n"):
            print(f"│ {line}")
        print(f"└{'─' * BOX_WIDTH}")
        print()


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    """Интерактивный CLI: ввод запроса → эксперимент с температурами."""
    print()
    print("=" * (BOX_WIDTH + 2))
    print("  День 4: Температура — сравнение temperature 0 / 0.7 / 1.2")
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

        total_calls = len(TEMPERATURES) * RUNS_PER_TEMP + 1  # +1 = AI judge
        print()
        print_progress(f"Всего {total_calls} вызовов DeepSeek API")

        # ── Фаза 1: эксперименты ──────────────────────────────────────────────
        print()
        by_temp = {}
        all_results = []

        for temp in TEMPERATURES:
            print(f"  [Temperature = {temp}]")
            by_temp[temp] = run_temperature(query, temp)
            for r in by_temp[temp]:
                all_results.append(r)
                status = "OK" if r["finish"] != "error" else "ERROR"
                print(f"    Run {r['run']}: {r['word_count']} слов, "
                      f"{r['duration']:.1f} с ({status})")
            print()

        # ── Фаза 2: AI-судья ──────────────────────────────────────────────────
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
                      "best_creativity": None, "best_consistency": None,
                      "recommendation": ""}
        print()

        # ── Фаза 3: показ всех ответов ───────────────────────────────────────
        # Спрашиваем, хочет ли пользователь увидеть ответы целиком
        print_progress("Показать все ответы? (Enter — да, n — пропустить)")
        show_all = input("  >>> ").strip().lower()
        if show_all != "n":
            print_all_responses_grouped(all_results)

        # ── Фаза 4: метрики ──────────────────────────────────────────────────
        print_metrics_table(by_temp, parsed)

        # ── Фаза 5: вердикт судьи ────────────────────────────────────────────
        print_header("Вердикт AI-судьи")
        for line in judge_result["judge_text"].strip().split("\n"):
            print(f"  {line}")
        print()

        # ── Фаза 6: рекомендации ─────────────────────────────────────────────
        print_recommendations(parsed)

        print_separator("=")
        print()
        print("  Можно ввести новый запрос или exit для выхода.")
        print()


if __name__ == "__main__":
    main()
