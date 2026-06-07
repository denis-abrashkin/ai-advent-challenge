"""
День 5. Версии моделей — сравнение weak, medium, strong моделей
================================================================
Запускает один и тот же промпт на трёх облачных моделях через
OpenRouter API, замеряет время ответа, количество токенов, стоимость,
сравнивает качество ответов, скорость и цену.

Модели:
  - Слабая:   google/gemma-3-4b-it                ($0.04 / $0.08 за 1M токенов)
  - Средняя:  meta-llama/llama-3.1-8b-instruct     ($0.02 / $0.03 за 1M токенов)
  - Сильная:  google/gemini-2.5-flash-lite         ($0.10 / $0.40 за 1M токенов)

API ключ читается из macOS Keychain (openrouter-api-key) или
переменной окружения OPENROUTER_API_KEY.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

# ── OpenRouter API ──────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Config ───────────────────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "google/gemma-3-4b-it",
        "label": "Слабая (Gemma 3 4B)",
        "tier": "weak",
        "provider": "Google",
        "price_per_m_input": 0.04,
        "price_per_m_output": 0.08,
    },
    {
        "name": "meta-llama/llama-3.1-8b-instruct",
        "label": "Средняя (Llama 3.1 8B)",
        "tier": "medium",
        "provider": "Meta",
        "price_per_m_input": 0.02,
        "price_per_m_output": 0.03,
    },
    {
        "name": "google/gemini-2.5-flash-lite",
        "label": "Сильная (Gemini 2.5 Flash Lite)",
        "tier": "strong",
        "provider": "Google",
        "price_per_m_input": 0.10,
        "price_per_m_output": 0.40,
    },
]

MODEL_LINKS = {
    m["name"]: f"https://openrouter.ai/models/{m['name']}"
    for m in MODELS
}

# Системный промпт (без эмодзи, как требует задание)
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely and accurately."
)


# ── Data classes ─────────────────────────────────────────────────────────────────


@dataclass
class ModelResult:  # pylint: disable=too-many-instance-attributes
    """Результат одного прогона модели для одного промпта."""
    model_name: str
    model_label: str
    model_tier: str
    provider: str
    prompt: str
    response: str
    duration_seconds: float
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_per_second: float = 0.0
    cost_usd: float = 0.0
    error: Optional[str] = None


# ── API key ──────────────────────────────────────────────────────────────────────


def _get_api_key() -> str:
    """Читает OpenRouter API key из macOS Keychain или переменной окружения."""
    # Пробуем macOS Keychain
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "openrouter-api-key", "-w"],
            capture_output=True, text=True, check=True,
        )
        key = result.stdout.strip()
        if key:
            return key
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: переменная окружения
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    sys.exit(
        "❌ API ключ OpenRouter не найден.\n\n"
        "Сохраните ключ в macOS Keychain:\n"
        "  security add-generic-password -s 'openrouter-api-key' -w 'sk-or-v1-...'\n\n"
        "Или установите переменную окружения:\n"
        "  export OPENROUTER_API_KEY='sk-or-v1-...'"
    )


def _get_openrouter_client():
    """Создаёт OpenAI-совместимый клиент для OpenRouter."""
    # Импортируем здесь для мягкого fallback
    try:
        from openai import OpenAI
    except ImportError as e:
        sys.exit(
            f"Ошибка импорта: {e}\n"
            "Установите openai:\n"
            "  pip install openai"
        )

    api_key = _get_api_key()
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/hyperion-vision/ai-advent-challenge",
            "X-Title": "AI Advent - Day 5 Model Comparison",
        },
    )


# ── Cost calculation ─────────────────────────────────────────────────────────────


def _calculate_cost(price_in, price_out, tokens_in, tokens_out) -> float:
    """Считает стоимость запроса в USD."""
    return (tokens_in / 1_000_000) * price_in + (tokens_out / 1_000_000) * price_out


# ── API call ─────────────────────────────────────────────────────────────────────


def call_model(
    client, model_id: str, prompt: str,
    max_tokens: Optional[int] = None, temperature: float = 0.7,
) -> tuple[str, int, int]:
    """
    Вызывает модель через OpenRouter API.
    max_tokens=None означает без ограничения длины ответа.
    Возвращает (текст ответа, входные токены, выходные токены).
    """
    kwargs = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None and max_tokens > 0:
        kwargs["max_tokens"] = max_tokens

    completion = client.chat.completions.create(**kwargs)
    response = completion.choices[0].message.content.strip() if completion.choices else ""
    usage = completion.usage
    return response, usage.prompt_tokens, usage.completion_tokens


# ── Judge ───────────────────────────────────────────────────────────────────────

JUDGE_MODEL = "google/gemini-2.5-flash-lite"
JUDGE_SYSTEM_PROMPT = (
    "You are an impartial AI judge evaluating responses from different language models. "
    "Assess each response critically and be fair. "
    "Return ALL text fields (strengths, weaknesses, summary, winner_reasoning) in Russian. "
    "Always return valid JSON."
)


@dataclass
class JudgeVerdict:
    """Вердикт судьи по одной модели."""
    model_name: str
    model_label: str
    scores: dict  # {"accuracy": 8, "completeness": 7, ...}
    total: float
    strengths: list[str]
    weaknesses: list[str]
    summary: str


def _build_judge_prompt(query: str, all_results: list) -> str:
    """Формирует промпт для судьи со всеми ответами моделей."""
    parts = [
        f'Исходный запрос пользователя: "{query}"\n',
        "Ниже приведены ответы трёх разных AI-моделей на один и тот же запрос. ",
        "Оцени каждый ответ по следующим критериям (оценка 1-10):\n",
        "1. **accuracy** — точность: насколько ответ фактологически корректен\n",
        "2. **completeness** — полнота: насколько глубоко раскрыта тема\n",
        "3. **clarity** — ясность: структурированность, читаемость, понятность\n",
        "4. **conciseness** — лаконичность: информативность без лишней воды\n\n",
        'Верни JSON-объект строго по этой схеме (без markdown-обёртки):\n',
        "{\n",
        '  "evaluations": [\n',
        "    {\n",
        '      "model": "model-id",\n',
        '      "label": "Model Label",\n',
        '      "scores": {"accuracy": 0, "completeness": 0, "clarity": 0, "conciseness": 0},\n',
        '      "overall": 0.0,\n',
        '      "strengths": ["сильная сторона на русском"],\n',
        '      "weaknesses": ["слабая сторона на русском"],\n',
        '      "summary": "вердикт одной строкой на русском"\n',
        "    }\n",
        "  ],\n",
        '  "winner": "best-model-id",\n',
        '  "winner_reasoning": "почему эта модель лучше — на русском"\n',
        "}\n",
        "\nВАЖНО: Все текстовые поля (strengths, weaknesses, summary, winner_reasoning) "
        "пиши ТОЛЬКО на РУССКОМ языке.\n",
        "\n--- ОТВЕТЫ МОДЕЛЕЙ ---\n",
    ]
    for r in all_results:
        if r.error:
            continue
        parts.append(f"\n### {r.model_label} ({r.model_name})\n{'-' * 40}\n")
        parts.append(r.response)
        parts.append("\n")
    return "\n".join(parts)


def _judge_responses(client, query: str, all_results: list) -> Optional[dict]:
    """Оценивает ответы всех моделей через LLM-судью."""
    successful = [r for r in all_results if not r.error]
    if len(successful) < 2:
        return None

    prompt = _build_judge_prompt(query, successful)

    for attempt in range(2):
        kwargs = {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        if attempt == 0:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = client.chat.completions.create(**kwargs)
            text = completion.choices[0].message.content.strip()
        except Exception:  # pylint: disable=broad-except
            continue

        # Strip optional markdown code fence
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue

        verdicts = []
        for ev in data.get("evaluations", []):
            scores = ev.get("scores", {})
            vals = [scores.get(k, 0) for k in ("accuracy", "completeness", "clarity", "conciseness")]
            total = sum(vals) / len(vals) if vals else 0.0
            verdicts.append(JudgeVerdict(
                model_name=ev.get("model", ""),
                model_label=ev.get("label", ev.get("model", "")),
                scores=scores,
                total=total,
                strengths=ev.get("strengths", []),
                weaknesses=ev.get("weaknesses", []),
                summary=ev.get("summary", ""),
            ))

        return {
            "verdicts": verdicts,
            "winner": data.get("winner", ""),
            "winner_reasoning": data.get("winner_reasoning", ""),
        }

    return None


def show_judge_results(judge_data: dict):
    """Показывает результаты оценки судьи в консоли."""
    if not judge_data:
        return
    verdicts = judge_data["verdicts"]

    col_width = 14
    parts = "  Критерий".ljust(20)
    for v in verdicts:
        parts += v.model_label.split("(")[0].strip().rjust(col_width)
    print(parts)
    print("  " + "─" * 20 + " " + " ".join("─" * col_width for _ in verdicts))

    for criterion, label in (("accuracy", "Точность"), ("completeness", "Полнота"),
                              ("clarity", "Ясность"), ("conciseness", "Лаконичность")):
        parts = f"  {label}".ljust(20)
        for v in verdicts:
            parts += str(v.scores.get(criterion, "—")).rjust(col_width)
        print(parts)

    parts = "  Общий балл".ljust(20)
    for v in verdicts:
        parts += f"{v.total:.1f}".rjust(col_width)
    print(parts)

    if judge_data.get("winner"):
        print(f"\n  🏆 Победитель: {judge_data['winner_reasoning']}")
    print()


# ── Вывод в консоль (русский, интерактивный) ────────────────────────────────────


BOX_WIDTH = 78


def print_separator(char="="):
    """Рисует горизонтальную линию."""
    print(char * (BOX_WIDTH + 2))


def print_header(title):
    """Заголовок секции."""
    print()
    print_separator("=")
    print(f"  {title}")
    print_separator("=")
    print()


def print_progress(msg):
    """Сообщение о ходе."""
    print(f"  >>> {msg}...")


def show_model_responses(all_results):
    """Показывает ответы всех моделей на запрос."""
    for model_cfg in MODELS:
        mn = model_cfg["name"]
        matches = [r for r in all_results if r.model_name == mn]
        if not matches:
            continue
        r = matches[0]

        print(f"┌─ {model_cfg['label']} ({model_cfg['provider']})")
        if r.error:
            print(f"│  [ОШИБКА] {r.error}")
        else:
            for line in r.response.split("\n"):
                print(f"│ {line}")
        print(f"└{'─' * BOX_WIDTH}")
        print()


def show_metrics_table(all_results):
    """Сводная таблица метрик по всем моделям."""
    col_width = 20

    # Заголовок
    parts = "  Параметр".ljust(22)
    for mc in MODELS:
        parts += mc["name"].rsplit("/", 1)[-1].rjust(col_width)
    print(parts)
    print("  " + "─" * 22 + " " + " ".join("─" * col_width for _ in MODELS))

    # Время
    parts = "  Время (с)".ljust(22)
    for mc in MODELS:
        matches = [r for r in all_results if r.model_name == mc["name"]]
        if matches and not matches[0].error:
            parts += f"{matches[0].duration_seconds:.1f}".rjust(col_width)
        else:
            parts += "—".rjust(col_width)
    print(parts)

    # Входные токены
    parts = "  Входных токенов".ljust(22)
    for mc in MODELS:
        matches = [r for r in all_results if r.model_name == mc["name"]]
        if matches and not matches[0].error:
            parts += str(matches[0].tokens_input).rjust(col_width)
        else:
            parts += "—".rjust(col_width)
    print(parts)

    # Выходные токены
    parts = "  Выходных токенов".ljust(22)
    for mc in MODELS:
        matches = [r for r in all_results if r.model_name == mc["name"]]
        if matches and not matches[0].error:
            parts += str(matches[0].tokens_output).rjust(col_width)
        else:
            parts += "—".rjust(col_width)
    print(parts)

    # Скорость
    parts = "  Скорость (ток/с)".ljust(22)
    for mc in MODELS:
        matches = [r for r in all_results if r.model_name == mc["name"]]
        if matches and not matches[0].error:
            parts += f"{matches[0].tokens_per_second:.1f}".rjust(col_width)
        else:
            parts += "—".rjust(col_width)
    print(parts)

    # Стоимость
    parts = "  Стоимость ($)".ljust(22)
    for mc in MODELS:
        matches = [r for r in all_results if r.model_name == mc["name"]]
        if matches and not matches[0].error:
            cost = matches[0].cost_usd
            if cost < 0.0001:
                parts += "< 0.0001".rjust(col_width)
            else:
                parts += f"{cost:.6f}".rjust(col_width)
        else:
            parts += "—".rjust(col_width)
    print(parts)


def _save_results_json(output_file, results, extra):
    """Сохраняет результаты в JSON-файл (дозаписывает в массив)."""
    if not output_file:
        return
    data = {**extra, "results": [asdict(r) for r in results]}
    existing = []
    if os.path.exists(output_file):
        try:
            with open(output_file, encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    existing = content
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(data)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"  💾 Результаты сохранены в {output_file}")


def _result_for_model(results, model_name):
    """Находит результат для указанной модели."""
    for r in results:
        if r.model_name == model_name:
            return r
    return None


def _save_markdown_report(output_file, results, extra, judge_data=None):
    """Сохраняет сравнительный отчёт в Markdown-файл."""
    if not output_file:
        return
    report_path = output_file.rsplit(".", 1)[0] + ".md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Сравнение моделей через OpenRouter\n\n")
        f.write("## Параметры запуска\n\n")
        f.write("- **Провайдер:** OpenRouter API\n")
        f.write(f"- **Температура:** {extra.get('temperature', '—')}\n")
        f.write(f"- **Макс. токенов:** {extra.get('max_tokens', '—')}\n")
        if extra.get("prompt"):
            f.write(f"- **Запрос:** {extra['prompt']}\n")
        f.write("\n")

        f.write("## Сравнительная таблица\n\n")
        # Header
        f.write("| Метрика |")
        for mc in MODELS:
            short = mc["name"].rsplit("/", 1)[-1]
            f.write(f" {short} |")
        f.write("\n|")
        f.write("---------|")
        for _ in MODELS:
            f.write(":----:|")
        f.write("\n")

        # Все строки таблицы
        for label, cell_fn in (
            ("Провайдер", lambda mc, _r: mc["provider"]),
            ("Время (с)", lambda _mc, r: f"{r.duration_seconds:.1f}"),
            ("Скорость (ток/с)", lambda _mc, r: f"{r.tokens_per_second:.1f}"),
            ("Входных токенов", lambda _mc, r: str(r.tokens_input)),
            ("Выходных токенов", lambda _mc, r: str(r.tokens_output)),
            ("Стоимость ($)", lambda _mc, r: (
                "< 0.0001" if r.cost_usd < 0.0001 else f"{r.cost_usd:.6f}"
            )),
        ):
            f.write(f"| {label} |")
            for mc in MODELS:
                r = _result_for_model(results, mc["name"])
                if r and not r.error:
                    f.write(f" {cell_fn(mc, r)} |")
                else:
                    f.write(" — |")
            f.write("\n")

        f.write("\n")
        f.write("## Цены моделей\n\n")
        f.write("| Модель | Вход ($/1M токенов) | Выход ($/1M токенов) |\n")
        f.write("|-------|:------------------:|:-------------------:|\n")
        for mc in MODELS:
            short = mc["name"].rsplit("/", 1)[-1]
            pin = f"{mc['price_per_m_input']:.2f}" if mc['price_per_m_input'] > 0 else "free"
            pout = f"{mc['price_per_m_output']:.2f}" if mc['price_per_m_output'] > 0 else "free"
            f.write(f"| {short} | {pin} | {pout} |\n")
        f.write("\n")

        # ── Judge section ───────────────────────────────────────────────────────
        if judge_data and judge_data.get("verdicts"):
            f.write("## Оценка судьи (LLM-as-a-Judge)\n\n")
            f.write(f"Модель-судья: `{extra.get('judge', {}).get('model', JUDGE_MODEL)}`\n\n")

            f.write("| Модель | Точность | Полнота | Ясность | Лаконичность | **Общий балл** |\n")
            f.write("|-------|:--------:|:-------:|:-------:|:------------:|:--------------:|\n")
            for v in judge_data["verdicts"]:
                short = v.model_name.rsplit("/", 1)[-1]
                s = v.scores
                f.write(
                    f"| {short} | {s.get('accuracy', '—')} "
                    f"| {s.get('completeness', '—')} "
                    f"| {s.get('clarity', '—')} "
                    f"| {s.get('conciseness', '—')} "
                    f"| **{v.total:.1f}** |\n"
                )
            f.write("\n")

            f.write("### Сильные и слабые стороны\n\n")
            for v in judge_data["verdicts"]:
                short = v.model_name.rsplit("/", 1)[-1]
                f.write(f"**{short}** ({v.total:.1f}/10)\n\n")
                if v.strengths:
                    f.write("✅ Сильные стороны:\n")
                    for s in v.strengths:
                        f.write(f"- {s}\n")
                if v.weaknesses:
                    f.write("❌ Слабые стороны:\n")
                    for s in v.weaknesses:
                        f.write(f"- {s}\n")
                f.write(f"\n*{v.summary}*\n\n")

            if judge_data.get("winner"):
                f.write("### Победитель\n\n")
                f.write(f"**{judge_data['winner']}**\n\n")
                f.write(f"{judge_data['winner_reasoning']}\n\n")

        f.write("## Ссылки на модели\n\n")
        for mc in MODELS:
            url = MODEL_LINKS.get(mc["name"], "")
            f.write(f"- [{mc['name'].rsplit('/', maxsplit=1)[-1]}]({url})\n")
        f.write("\n")

    print(f"  📊 Отчёт сохранён в {report_path}")


def _slugify(text, max_words=5):
    """Generate a short URL-safe slug from the first words of text."""
    words = text.split()[:max_words]
    raw = " ".join(words)
    clean = re.sub(r"[^a-zA-Z0-9 ]", "", raw).strip().lower()
    slug = "-".join(clean.split())
    return slug[:60] or "query"


def _next_run_id(base_dir):
    """Find next run ID from existing numbered folders (NNN-*)."""
    if not os.path.isdir(base_dir):
        return 1
    max_id = 0
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if os.path.isdir(entry_path):
            parts = entry.split("-", 1)
            if parts[0].isdigit():
                max_id = max(max_id, int(parts[0]))
    return max_id + 1


def _resolve_output_file(output_target, query):
    """Resolve output file path: explicit file or auto-save subfolder."""
    if not output_target:
        return None
    # Explicit file path (.json / .md given by user)
    if output_target.endswith((".json", ".md")):
        return output_target
    # Auto-save mode: create numbered subfolder per query
    os.makedirs(output_target, exist_ok=True)
    run_id = _next_run_id(output_target)
    slug = _slugify(query)
    run_dir = os.path.join(output_target, f"{run_id:03d}-{slug}")
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "results.json")


def process_single_query(query, max_tokens, temperature, output_file):  # pylint: disable=too-many-locals
    """Прогоняет один запрос через все модели и показывает результат."""
    print()
    print_progress(f"Запрос: {query[:80]}")
    print()

    client = _get_openrouter_client()

    all_results = []
    for model_cfg in MODELS:
        print(f"  [Запрос] {model_cfg['label']}...", end=" ", flush=True)
        t0 = time.time()
        try:
            response, inp_len, out_len = call_model(
                client, model_cfg["name"], query,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.time() - t0
            tok_per_sec = out_len / elapsed if elapsed > 0 else 0.0
            cost = _calculate_cost(
                model_cfg["price_per_m_input"],
                model_cfg["price_per_m_output"],
                inp_len,
                out_len,
            )

            all_results.append(ModelResult(
                model_name=model_cfg["name"],
                model_label=model_cfg["label"],
                model_tier=model_cfg["tier"],
                provider=model_cfg["provider"],
                prompt=query,
                response=response,
                duration_seconds=elapsed,
                tokens_input=inp_len,
                tokens_output=out_len,
                tokens_per_second=tok_per_sec,
                cost_usd=cost,
            ))

            if cost < 0.0001:
                cost_str = "бесплатно"
            else:
                cost_str = f"${cost:.6f}"
            print(f"✅ {out_len} токенов, {elapsed:.1f}с ({tok_per_sec:.1f} ток/с), {cost_str}")

        except Exception as e:  # pylint: disable=broad-except
            elapsed = time.time() - t0
            all_results.append(ModelResult(
                model_name=model_cfg["name"],
                model_label=model_cfg["label"],
                model_tier=model_cfg["tier"],
                provider=model_cfg["provider"],
                prompt=query,
                response="",
                duration_seconds=elapsed,
                error=str(e),
            ))
            print(f"❌ ОШИБКА — {e}")

    print()
    extra = {
        "provider": "OpenRouter",
        "prompt": query,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # ── Оценка судьи ────────────────────────────────────────────────────────────
    judge_data = None
    successful = [r for r in all_results if not r.error]
    if len(successful) >= 2:
        print_progress("Оценка ответов судьёй (LLM-as-a-Judge)")
        judge_data = _judge_responses(client, query, all_results)
        if judge_data:
            extra["judge"] = {
                "model": JUDGE_MODEL,
                "winner": judge_data["winner"],
                "winner_reasoning": judge_data["winner_reasoning"],
                "verdicts": [asdict(v) for v in judge_data["verdicts"]],
            }

    _save_results_json(output_file, all_results, extra)
    _save_markdown_report(output_file, all_results, extra, judge_data)
    print_header("Ответы моделей")
    show_model_responses(all_results)
    print_header("Сравнение метрик")
    show_metrics_table(all_results)
    if judge_data:
        print_header("Оценка судьи (LLM-as-a-Judge)")
        show_judge_results(judge_data)
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────────


def main():
    """Точка входа: интерактивный режим или разовый запрос."""
    parser = argparse.ArgumentParser(
        description="День 5: Сравнение слабой / средней / сильной модели через OpenRouter API"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Одиночный запрос (без интерактивного режима)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Максимум токенов в ответе (0 = без ограничения)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Температура семплинга (по умолчанию: 0.7)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Сохранить результаты в указанный файл (по умолчанию: output/ID-тема/)",
    )
    args = parser.parse_args()

    # Определяем цель вывода: явный файл или авто-сохранение в output/
    output_target = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output"
    )

    # Режим: одиночный запрос (без интерактива)
    if args.prompt:
        output_file = _resolve_output_file(output_target, args.prompt)
        process_single_query(
            args.prompt, args.max_tokens, args.temperature, output_file
        )
        return

    # ── Интерактивный режим ────────────────────────────────────────────────────
    print()
    print("=" * (BOX_WIDTH + 2))
    print("  День 5: Версии моделей — сравнение слабой / средней / сильной")
    print("=" * (BOX_WIDTH + 2))
    print()
    print("  Провайдер: OpenRouter API")
    print("  Модели:")
    for mc in MODELS:
        price = f"${mc['price_per_m_input']:.2f}" if mc['price_per_m_input'] > 0 else "free"
        print(f"    • {mc['label']} — {mc['name']} ({price}/1M in)")
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

        output_file = _resolve_output_file(output_target, query)
        process_single_query(
            query, args.max_tokens, args.temperature, output_file
        )


if __name__ == "__main__":
    main()
