"""
День 3. Разные способы рассуждения
===================================
Одна задача — четыре способа решения через DeepSeek API:

  1. Прямой ответ (baseline) — без дополнительных инструкций
  2. Пошаговое решение — инструкция «решай по шагам»
  3. Авто-промпт — модель генерирует промпт, затем решает по нему
  4. Команда экспертов — аналитик, инженер и критик (3 независимых вызова)

Сравнение:
  - Эвристическая проверка по ключевым словам
  - Оценка AI-судьёй (отдельный вызов API)
  - Сводная таблица с метриками
"""

import os
import textwrap
import time
from openai import OpenAI

# ── Настройка ──────────────────────────────────────────────────────────────────

api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
if not api_key:
    raise RuntimeError(
        "ANTHROPIC_AUTH_TOKEN не найден — проверьте экспорт переменной"
    )

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

STOP_MARKER = "[DONE]"

SYSTEM_BASE = "You are a helpful assistant. Do not use emojis in your responses."

# ── Задача ─────────────────────────────────────────────────────────────────────

TASK = textwrap.dedent("""\
    Загадка пропавшего доллара (Missing Dollar Riddle):

    Трое человек заселяются в отель. Номер стоит $30.
    Каждый платит по $10. Менеджер понимает, что номер
    должен стоить $25, и даёт $5 горничной, чтобы та
    вернула гостям. Горничная забирает $2 себе, а каждому
    гостю возвращает по $1.

    Вопрос: каждый гость заплатил $9 ($10 - $1), итого $27.
    Горничная оставила $2. 27 + 2 = 29. Где пропавший доллар?

    Объясни, в чём ошибка рассуждения и куда пропал доллар.
""")

GROUND_TRUTH = textwrap.dedent("""\
    Доллар никуда не пропадал. Ошибка — в некорректном сложении.

    $27, заплаченные гостями, УЖЕ включают $2 горничной:
      $25 (отель) + $2 (горничная) = $27.

    Прибавлять $2 горничной к $27 — значит учитывать их дважды.

    Правильный счёт:
      $25 (отель) + $2 (горничная) + $3 (сдача гостям) = $30.
""")

# ── Роли экспертов ─────────────────────────────────────────────────────────────

EXPERT_ROLES = {
    "analyst": textwrap.dedent("""\
        Вы — математический аналитик. Ваша специализация — формальный
        разбор логических парадоксов.

        Подход к задаче:
        1. Запишите все денежные потоки в виде уравнений.
        2. Проверьте каждое утверждение из условия на корректность.
        3. Покажите, какая операция в рассуждении «27 + 2 = 29» некорректна.
        4. Приведите правильный баланс.

        Отвечайте строго, без риторических отступлений.
    """),
    "engineer": textwrap.dedent("""\
        Вы — инженер-практик. Ваша сила — понятные объяснения на
        простых примерах, без лишней теории.

        Подход к задаче:
        1. Объясните суть ошибки на бытовом примере.
        2. Покажите, как правильно посчитать деньги.
        3. Дайте короткий, запоминающийся вывод.

        Отвечайте просто и по делу.
    """),
    "critic": textwrap.dedent("""\
        Вы — логический критик. Ваша задача — найти и разоблачить
        ошибку в представленном рассуждении.

        Подход к задаче:
        1. Найдите конкретное место в цепочке «$30 → $27 → $29»,
           где происходит подмена смысла.
        2. Объясните, почему операция 27 + 2 логически бессмысленна.
        3. Предложите корректный способ проверки.

        Будьте придирчивы к каждой детали рассуждения.
    """),
}

# ── Вызов API ──────────────────────────────────────────────────────────────────


def ask(messages, **kwargs):
    """Отправляет запрос к DeepSeek и возвращает текст ответа + finish_reason."""
    resp = client.chat.completions.create(
        model="deepseek-chat", messages=messages, **kwargs
    )
    choice = resp.choices[0]
    return choice.message.content or "", choice.finish_reason


# ── Вспомогательные структуры ──────────────────────────────────────────────────


def make_result(title, method_key, params, text, finish, duration,
                steps=1, intermediate=None):
    """Создаёт стандартизированный словарь результата."""
    return {
        "title": title,
        "method": method_key,
        "params": params,
        "text": text.strip(),
        "finish": finish,
        "duration": duration,
        "steps": steps,
        "char_count": len(text),
        "word_count": len(text.split()) if text else 0,
        "intermediate": intermediate,
        "criteria_score": None,
        "ai_score": None,
    }


# ── Метод 1: Прямой ответ ─────────────────────────────────────────────────────


def solve_direct(task):
    """
    Метод 1: Прямой ответ без дополнительных инструкций.
    System: базовая инструкция ассистента.
    User: текст задачи.
    """
    t0 = time.time()
    text, finish = ask(
        messages=[
            {"role": "system", "content": SYSTEM_BASE},
            {"role": "user", "content": task},
        ],
    )
    duration = time.time() - t0

    return make_result(
        title="1. Прямой ответ (baseline)",
        method_key="direct",
        params="Без дополнительных инструкций",
        text=text,
        finish=finish,
        duration=duration,
        steps=1,
    )


# ── Метод 2: Пошаговое решение ────────────────────────────────────────────────


def solve_step_by_step(task):
    """
    Метод 2: Инструкция «решай по шагам».
    System: базовая инструкция + требование пошагового рассуждения.
    User: текст задачи.
    """
    system_prompt = textwrap.dedent(f"""\
        {SYSTEM_BASE}

        Решай задачу по шагам. Распиши каждый шаг рассуждения
        отдельно, нумеруя шаги. Проверяй каждое арифметическое
        действие. В конце напиши итоговый ответ.

        После завершения напиши '{STOP_MARKER}'.
    """)

    t0 = time.time()
    text, finish = ask(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ],
    )
    duration = time.time() - t0

    return make_result(
        title="2. Пошаговое решение",
        method_key="step_by_step",
        params="Инструкция «решай по шагам» в system prompt",
        text=text,
        finish=finish,
        duration=duration,
        steps=1,
    )


# ── Метод 3: Авто-промпт ─────────────────────────────────────────────────────


def solve_auto_prompt(task):
    """
    Метод 3: Модель генерирует промпт для решения, затем решает по нему.

    Шаг A — промпт-инжиниринг:
        System: инструкция эксперта по промптам.
        User: текст задачи.
        → получаем сгенерированный промпт.

    Шаг B — решение:
        System: сгенерированный промпт.
        User: текст задачи.
        → получаем ответ.
    """
    prompt_gen_system = textwrap.dedent("""\
        Ты — эксперт по промпт-инжинирингу.
        Твоя задача: составить эффективный системный промпт для
        решения логической задачи, которую передаст пользователь.

        Промпт должен направлять модель к:
        - Внимательному анализу условия
        - Проверке каждого арифметического действия
        - Поиску логической ошибки в рассуждении
        - Чёткому объяснению правильного ответа

        Напиши ТОЛЬКО текст промпта. Без «Вот промпт:», без кавычек,
        без пояснений — только готовый к использованию system prompt.
    """)

    # Шаг A: генерация промпта
    t0 = time.time()
    generated_prompt, _ = ask(
        messages=[
            {"role": "system", "content": prompt_gen_system},
            {"role": "user", "content": task},
        ],
    )

    # Шаг B: решение сгенерированным промптом
    text, finish = ask(
        messages=[
            {"role": "system", "content": generated_prompt},
            {"role": "user", "content": task},
        ],
    )
    duration = time.time() - t0

    return make_result(
        title="3. Авто-промпт",
        method_key="auto_prompt",
        params="Генерация промпта → решение (2 вызова API)",
        text=text,
        finish=finish,
        duration=duration,
        steps=2,
        intermediate=generated_prompt,
    )


# ── Метод 4: Команда экспертов ────────────────────────────────────────────────


def solve_experts(task):
    """
    Метод 4: Три эксперта (аналитик, инженер, критик).
    Каждый получает свою роль в system prompt.
    Три независимых вызова API.
    """
    experts_data = {}
    total_duration = 0.0

    for role_key, role_desc in EXPERT_ROLES.items():
        system_prompt = f"{SYSTEM_BASE}\n\n{role_desc}"

        t0 = time.time()
        text, finish = ask(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
        )
        duration = time.time() - t0
        total_duration += duration

        experts_data[role_key] = {
            "text": text,
            "finish": finish,
            "duration": duration,
        }

    # Собираем все три ответа в один текст для вывода
    role_labels = {"analyst": "Аналитик", "engineer": "Инженер", "critic": "Критик"}
    combined_parts = []
    for key in ["analyst", "engineer", "critic"]:
        combined_parts.append(f"─── {role_labels[key]} ───")
        combined_parts.append(experts_data[key]["text"])
        combined_parts.append("")
    combined_text = "\n".join(combined_parts).strip()

    finish_summary = ", ".join(
        f"{role_labels[k]}: {experts_data[k]['finish']}"
        for k in ["analyst", "engineer", "critic"]
    )

    return make_result(
        title="4. Команда экспертов",
        method_key="experts",
        params="Аналитик + Инженер + Критик (3 независимых вызова)",
        text=combined_text,
        finish=finish_summary,
        duration=total_duration,
        steps=3,
    )


# ── Эвристическая проверка ─────────────────────────────────────────────────────


VALIDATION_CRITERIA = {
    "identifies_fallacy": [
        "нельзя складывать", "не имеет смысла", "категориальн",
        "ошибка", "заблуждение", "некорректн", "неправильн",
        "пропавшего доллара нет", "нет пропавшего", "никуда не пропадал",
        "никакого доллара", "нет никакого доллара", "не пропадал",
        "не существует пропавшего", "никуда не делся",
    ],
    "correct_accounting": [
        "25 + 2 + 3", "25 + 2 = 27", "27 + 3 = 30",
        "25+2+3", "25+2=27",
        "отель 25", "горничная 2", "сдача 3",
        "уже включен", "уже включён", "уже входят",
        "уже содержит", "уже учтен",
    ],
    "explains_clearly": [
        "правильный счёт", "правильное объяснение",
        "правильный баланс", "корректный баланс",
        "дважды", "повторно", "задвоен",
        "не надо прибавлять", "не нужно складывать",
        "не следует прибавлять",
    ],
}


def check_criteria(text):
    """
    Эвристическая проверка ответа по ключевым словам.
    Возвращает словарь с результатами по каждому критерию и общий счёт (0–3).
    """
    text_lower = text.lower()
    hits = {}
    score = 0
    for criterion, keywords in VALIDATION_CRITERIA.items():
        found = any(kw in text_lower for kw in keywords)
        hits[criterion] = found
        if found:
            score += 1
    return {"hits": hits, "score": score}


# ── Оценка AI-судьёй ───────────────────────────────────────────────────────────


def evaluate_answers(results, task):
    """
    Пятый вызов API: AI-судья оценивает все четыре ответа.

    System: инструкция судьи.
    User: задача + четыре ответа с метками методов.

    Возвращает текст вердикта.
    """
    judge_system = textwrap.dedent("""\
        Ты — беспристрастный судья. Твоя задача — оценить четыре ответа
        на логическую задачу «Missing Dollar Riddle».

        Критерии оценки (каждый от 1 до 10):
        1. Точность — правильно ли определена ошибка в рассуждении?
        2. Ясность — насколько понятно объяснение?
        3. Полнота — раскрыты ли все детали (куда делись деньги,
           почему 27+2=29 некорректно, каков правильный баланс)?

        Для каждого метода выведи ТРИ числа и средний балл.
        Затем назови лучший метод и обоснуй выбор.

        Формат вывода (строго):

        Метод 1 (Прямой ответ):
          Точность: X/10, Ясность: Y/10, Полнота: Z/10
          Среднее: A/10

        Метод 2 (Пошаговое решение):
          Точность: X/10, Ясность: Y/10, Полнота: Z/10
          Среднее: A/10

        Метод 3 (Авто-промпт):
          Точность: X/10, Ясность: Y/10, Полнота: Z/10
          Среднее: A/10

        Метод 4 (Команда экспертов):
          Точность: X/10, Ясность: Y/10, Полнота: Z/10
          Среднее: A/10

        === Вердикт ===
        Лучший метод: Метод N (название)
        Обоснование: <одно-два предложения>
    """)

    # Собираем текст для судьи
    parts = ["=== ЗАДАЧА ===\n" + task + "\n"]
    for r in results:
        parts.append(f"=== {r['title']} ===\n{r['text']}\n")
    judge_input = "\n".join(parts)

    text, finish = ask(
        messages=[
            {"role": "system", "content": judge_system},
            {"role": "user", "content": judge_input},
        ],
    )

    return {"judge_text": text, "finish": finish}


# ── Разбор оценок судьи ────────────────────────────────────────────────────────


def parse_judge_scores(judge_text):
    """
    Извлекает средние баллы из текста вердикта судьи.
    Ищет строки вида «Среднее: X.Y/10» после каждого метода.
    Возвращает словарь {method_num: float_score} и номер лучшего метода.
    """
    import re

    scores = {}
    lines = judge_text.split("\n")

    current_method = None
    for line in lines:
        # Ищем строку «Метод N ...»
        for i in range(1, 5):
            if f"Метод {i}" in line and current_method is None:
                current_method = i
                break
        else:
            # Ищем «Среднее: X.Y/10» или «Среднее: X/10»
            if "Среднее" in line and current_method is not None and current_method <= 4:
                match = re.search(r"Среднее:\s*(\d+[.,]?\d*)\s*/?\s*\d*", line)
                if match:
                    score_str = match.group(1).replace(",", ".")
                    scores[current_method] = float(score_str)
                current_method = None
                continue

    # Ищем лучший метод
    best_method = None
    for line in lines:
        if "Лучший метод" in line or "лучший метод" in line:
            for i in range(1, 5):
                if f"Метод {i}" in line:
                    best_method = i
                    break
            break

    return {"scores": scores, "best_method": best_method}


# ── Вывод ──────────────────────────────────────────────────────────────────────


BOX_WIDTH = 78


def print_separator(char="="):
    """Горизонтальная линия-разделитель."""
    print(char * (BOX_WIDTH + 2))


def print_header(title):
    """Заголовок секции."""
    print()
    print_separator("=")
    print(f"  {title}")
    print_separator("=")
    print()


def print_task_block(task, ground_truth):
    """Выводит условие задачи и правильный ответ."""
    print_header("Задача дня: Пропавший доллар")

    print("┌─ Условие")
    for line in task.strip().split("\n"):
        print(f"│ {line}")
    print("└" + "─" * BOX_WIDTH)

    print()
    print("┌─ Правильный ответ (для сравнения)")
    for line in ground_truth.strip().split("\n"):
        print(f"│ {line}")
    print("└" + "─" * BOX_WIDTH)
    print()


def print_progress(label):
    """Сообщение о ходе выполнения."""
    print(f"  >>> {label}...")


def print_result(result, show_intermediate=False):
    """Выводит один результат в формате Unicode-бокса."""
    title = result["title"]
    print(f"┌─ {title}")
    print(f"│ Параметры: {result['params']}")
    print(f"│ Шагов API: {result['steps']}")
    print(f"│ Время: {result['duration']:.2f} с")
    print(f"│ Символов: {result['char_count']} │ Слов: {result['word_count']}")
    print(f"│ Finish: {result['finish']}")
    if result.get("criteria_score") is not None:
        print(f"│ Критериев пройдено: {result['criteria_score']}/3")
    print("│")

    if show_intermediate and result.get("intermediate"):
        print("│ ┌─ Сгенерированный промпт:")
        for line in result["intermediate"].strip().split("\n"):
            print(f"│ │ {line}")
        print("│ └" + "─" * (BOX_WIDTH - 2))
        print("│")

    for line in result["text"].split("\n"):
        print(f"│ {line}")
    print("└" + "─" * BOX_WIDTH)
    print()


def print_comparison_table(results):
    """Сводная таблица сравнения всех методов."""
    print_header("Сравнение методов")

    # Заголовки и ширины колонок
    headers = ["Метод", "API", "Симв.", "Слов", "Время", "Крит.", "AI"]
    col_widths = [24, 4, 7, 6, 9, 6, 5]

    # Верхняя граница
    top = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    print(top)

    # Заголовки
    cells = []
    for h, w in zip(headers, col_widths):
        cells.append(f" {h:^{w}}")
    print("│" + "│".join(cells) + "│")

    # Разделитель
    sep = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    print(sep)

    # Данные
    for r in results:
        short_title = r["title"].split("(")[0].strip().replace(
            "1. ", ""
        ).replace("2. ", "").replace("3. ", "").replace("4. ", "")
        # Обрезаем название до ширины колонки
        if len(short_title) > 23:
            short_title = short_title[:22] + "…"

        ai_score = r.get("ai_score")
        if ai_score is not None:
            ai_str = f"{ai_score:.1f}"
        else:
            ai_str = "?"

        row = [
            f" {short_title:<{col_widths[0] - 1}}",
            f" {r['steps']:>2} ",
            f" {r['char_count']:>5} ",
            f" {r['word_count']:>4} ",
            f" {r['duration']:>5.1f} с ",
            f"  {r.get('criteria_score', '?'):>1}/3  ",
            f"  {ai_str:>4} ",
        ]
        print("│" + "│".join(row) + "│")

    # Нижняя граница
    bottom = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
    print(bottom)


def print_ai_evaluation(ai_eval):
    """Выводит текст вердикта AI-судьи."""
    print_header("Вердикт AI-судьи")
    for line in ai_eval["judge_text"].strip().split("\n"):
        print(f"  {line}")
    print()


def print_ground_truth():
    """Выводит правильный ответ."""
    print_header("Правильный ответ (эталон)")
    for line in GROUND_TRUTH.strip().split("\n"):
        print(f"  {line}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    """Запускает все четыре метода, сравнивает и выводит результаты."""
    print()
    print_separator("=")
    print("  День 3: Разные способы рассуждения")
    print("  Missing Dollar Riddle — 4 способа, 1 задача")
    print_separator("=")
    print()

    # Показываем задачу
    print_task_block(TASK, GROUND_TRUTH)

    input("  Нажмите Enter, чтобы запустить решение (8 вызовов DeepSeek API)... ")
    print()

    # ── Запуск четырёх методов ──────────────────────────────────────────────

    methods = [
        ("Прямой ответ", solve_direct),
        ("Пошаговое решение", solve_step_by_step),
        ("Авто-промпт", solve_auto_prompt),
        ("Команда экспертов", solve_experts),
    ]

    results = []
    for i, (name, func) in enumerate(methods, 1):
        print_progress(f"[{i}/4] {name}")
        try:
            result = func(TASK)
            # Эвристическая проверка
            criteria = check_criteria(result["text"])
            result["criteria_score"] = criteria["score"]
            result["criteria_hits"] = criteria["hits"]
            results.append(result)
            print(f"       Готово (критериев: {criteria['score']}/3, "
                  f"время: {result['duration']:.2f} с)")
        except Exception as e:
            print(f"       ОШИБКА: {e}")
            results.append(make_result(
                title=f"ОШИБКА: {name}",
                method_key="error",
                params=str(e),
                text=f"Не удалось получить ответ: {e}",
                finish="error",
                duration=0,
            ))
        print()

    # ── Поочерёдный показ результатов ───────────────────────────────────────

    print_separator("-")
    print("  Результаты (нажмите Enter для перехода к следующему)")
    print_separator("-")
    print()

    for i, result in enumerate(results):
        print_result(result, show_intermediate=(result["method"] == "auto_prompt"))
        if i < len(results) - 1:
            input("  Нажмите Enter для следующего результата... ")
            print()

    # ── Оценка AI-судьёй ────────────────────────────────────────────────────

    print()
    print_progress("Оценка AI-судьёй (5-й вызов API)")
    try:
        ai_eval = evaluate_answers(results, TASK)
        print("       Готово")
    except Exception as e:
        print(f"       ОШИБКА при оценке: {e}")
        ai_eval = {"judge_text": f"Оценка не удалась: {e}", "finish": "error"}

    # Извлекаем баллы из вердикта и добавляем в результаты
    parsed = parse_judge_scores(ai_eval["judge_text"])
    for result in results:
        method_num = results.index(result) + 1
        result["ai_score"] = parsed["scores"].get(method_num)

    # ── Сводная таблица ─────────────────────────────────────────────────────

    print_comparison_table(results)

    # ── Вердикт судьи ───────────────────────────────────────────────────────

    print_ai_evaluation(ai_eval)

    # ── Эталон ──────────────────────────────────────────────────────────────

    print_ground_truth()

    # ── Итоги ────────────────────────────────────────────────────────────────

    print_header("Итоги")

    best_from_ai = parsed.get("best_method")
    if best_from_ai:
        method_names = {
            1: "Прямой ответ",
            2: "Пошаговое решение",
            3: "Авто-промпт",
            4: "Команда экспертов",
        }
        print(f"  Лучший метод по версии AI-судьи: {best_from_ai}. "
              f"{method_names.get(best_from_ai, '')}")

    # Находим метод с максимальным эвристическим счётом
    max_criteria = max(
        (r for r in results if r.get("criteria_score") is not None),
        key=lambda r: r["criteria_score"],
        default=None,
    )
    if max_criteria:
        print(f"  Лучший метод по эвристике (ключевые слова): "
              f"{max_criteria['criteria_score']}/3 — "
              f"{max_criteria['title'].split('(')[0].strip()}")

    # Самый быстрый
    fastest = min(results, key=lambda r: r["duration"])
    print(f"  Самый быстрый: {fastest['duration']:.2f} с — "
          f"{fastest['title'].split('(')[0].strip()}")

    print()
    print_separator("=")
    print()


if __name__ == "__main__":
    main()
