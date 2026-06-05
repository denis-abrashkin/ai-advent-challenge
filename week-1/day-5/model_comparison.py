"""
День 5. Версии моделей — сравнение weak, medium, strong моделей
================================================================
Запускает один и тот же промпт на трёх моделях из HuggingFace,
замеряет время ответа, количество токенов, стоимость (если API),
сравнивает качество ответов, скорость и ресурсоёмкость.

Модели (семейство Qwen2.5 — одинаковый токенизатор и архитектура):
  - Слабая:   Qwen/Qwen2.5-0.5B-Instruct  (0.5B,  ~350 MB в fp16)
  - Средняя:  Qwen/Qwen2.5-1.5B-Instruct  (1.5B,  ~1.0 GB в fp16)
  - Сильная:  Qwen/Qwen2.5-3B-Instruct    (3B,    ~2.0 GB в fp16)

При недоступности MPS использует CPU.
При недостатке памяти для 7B использует 4-битную квантизацию.
"""

import argparse
import contextlib
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

# ── Try imports ──────────────────────────────────────────────────────────────────

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
except ImportError as e:
    sys.exit(
        f"Ошибка импорта: {e}\n"
        "Установите зависимости:\n"
        "  pip install transformers torch accelerate sentencepiece"
    )

# ── Config ───────────────────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Слабая (Weak, 0.5B)",
        "tier": "weak",
        "params_b": 0.5,
    },
    {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Средняя (Medium, 1.5B)",
        "tier": "medium",
        "params_b": 1.5,
    },
    {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "label": "Сильная (Strong, 3B)",
        "tier": "strong",
        "params_b": 3.0,
    },
]

# Ссылки на модели в HuggingFace
MODEL_LINKS = {
    model["name"]: f"https://huggingface.co/{model['name']}"
    for model in MODELS
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
    params_b: float
    prompt: str
    response: str
    device: str
    duration_seconds: float
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_per_second: float = 0.0
    error: Optional[str] = None
    quantization: str = "none"


# ── Device detection ─────────────────────────────────────────────────────────────


def detect_device() -> str:
    """Определяет лучшее доступное устройство."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Model loading ────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def managed_model(model_id: str, device: str):
    """
    Загружает модель и токенизатор, гарантированно выгружая их при выходе.
    Для 7B модели использует fp16 на MPS/cuda для экономии памяти.
    """
    model = None
    tokenizer = None
    quantization = "none"
    try:
        print(f"  [Загрузка] {model_id}...", end=" ", flush=True)
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Определяем dtype и устройство
        load_kwargs = {
            "trust_remote_code": True,
        }

        if device in ("mps", "cuda"):
            load_kwargs["dtype"] = torch.float16
            quantization = "fp16"
        else:
            load_kwargs["dtype"] = torch.float32
            quantization = "fp32"

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()

        # Явно перемещаем модель на нужное устройство
        if device in ("mps", "cuda"):
            model = model.to(device)

        elapsed = time.time() - t0
        print(f"готово ({elapsed:.1f}с, {quantization})")
        yield model, tokenizer, quantization

    finally:
        del model
        del tokenizer
        gc.collect()
        if device == "mps":
            with contextlib.suppress(RuntimeError):
                torch.mps.empty_cache()


# ── Inference ────────────────────────────────────────────────────────────────────


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> tuple[str, int, int]:
    """
    Выполняет инференс модели на заданном промпте.
    Возвращает (текст ответа, количество входных токенов, количество выходных токенов).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Применяем chat template
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_len = outputs.shape[1] - input_len
    response = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    return response, input_len, output_len


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

        print(f"┌─ {model_cfg['label']} ({model_cfg['params_b']}B)")
        if r.error:
            print(f"│  [ОШИБКА] {r.error}")
        else:
            for line in r.response.split("\n"):
                print(f"│ {line}")
        print(f"└{'─' * BOX_WIDTH}")
        print()


def show_metrics_table(all_results):
    """Сводная таблица метрик по всем моделям."""
    col_width = 16

    # Заголовок
    parts = "  Параметр".ljust(22)
    for mc in MODELS:
        parts += f"{mc['params_b']}B".rjust(col_width)
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

    # Токены
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


def _save_markdown_report(output_file, results, extra):
    """Сохраняет сравнительный отчёт в Markdown-файл."""
    if not output_file:
        return
    report_path = output_file.rsplit(".", 1)[0] + ".md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Сравнение моделей Qwen2.5\n\n")
        f.write("## Параметры запуска\n\n")
        f.write(f"- **Устройство:** {extra.get('device', '—')}\n")
        f.write(f"- **Температура:** {extra.get('temperature', '—')}\n")
        f.write(f"- **Макс. токенов:** {extra.get('max_tokens', '—')}\n")
        if extra.get("prompt"):
            f.write(f"- **Запрос:** {extra['prompt']}\n")
        f.write("\n")

        f.write("## Сравнительная таблица\n\n")
        # Header
        f.write("| Метрика |")
        for mc in MODELS:
            f.write(f" {mc['params_b']}B |")
        f.write("\n|")
        f.write("---------|")
        for _ in MODELS:
            f.write(":----:|")
        f.write("\n")

        # Строки таблицы
        for label, attr, fmt in (
            ("Время (с)", "duration_seconds", "{:.1f}"),
            ("Входных токенов", "tokens_input", "{}"),
            ("Выходных токенов", "tokens_output", "{}"),
            ("Скорость (ток/с)", "tokens_per_second", "{:.1f}"),
        ):
            f.write(f"| {label} |")
            for mc in MODELS:
                r = _result_for_model(results, mc["name"])
                if r and not r.error:
                    f.write(f" {fmt.format(getattr(r, attr))} |")
                else:
                    f.write(" — |")
            f.write("\n")

        f.write("\n")
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


def process_single_query(query, device, max_tokens, temperature, output_file):  # pylint: disable=too-many-locals
    """Прогоняет один запрос через все модели и показывает результат."""
    print()
    print_progress(f"Запрос: {query[:80]}")
    print()

    all_results = []
    for model_cfg in MODELS:
        with managed_model(model_cfg["name"], device) as (model, tokenizer, quant):
            if model is None:
                all_results.append(ModelResult(
                    model_name=model_cfg["name"],
                    model_label=model_cfg["label"],
                    model_tier=model_cfg["tier"],
                    params_b=model_cfg["params_b"],
                    prompt=query,
                    response="",
                    device=device,
                    duration_seconds=0.0,
                    error="Ошибка загрузки модели",
                    quantization=quant,
                ))
                continue

            t0 = time.time()
            try:
                response, inp_len, out_len = run_inference(
                    model, tokenizer, query,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                elapsed = time.time() - t0
                tok_per_sec = out_len / elapsed if elapsed > 0 else 0.0

                all_results.append(ModelResult(
                    model_name=model_cfg["name"],
                    model_label=model_cfg["label"],
                    model_tier=model_cfg["tier"],
                    params_b=model_cfg["params_b"],
                    prompt=query,
                    response=response,
                    device=device,
                    duration_seconds=elapsed,
                    tokens_input=inp_len,
                    tokens_output=out_len,
                    tokens_per_second=tok_per_sec,
                    quantization=quant,
                ))
                print(f"  ✅ {model_cfg['label']}: {out_len} токенов, "
                      f"{elapsed:.1f}с ({tok_per_sec:.1f} ток/с)")
            except (RuntimeError, ValueError, OSError) as e:
                elapsed = time.time() - t0
                all_results.append(ModelResult(
                    model_name=model_cfg["name"],
                    model_label=model_cfg["label"],
                    model_tier=model_cfg["tier"],
                    params_b=model_cfg["params_b"],
                    prompt=query,
                    response="",
                    device=device,
                    duration_seconds=elapsed,
                    error=str(e),
                    quantization=quant,
                ))
                print(f"  ❌ {model_cfg['label']}: ОШИБКА — {e}")

    print()
    extra = {
        "device": device, "prompt": query,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    _save_results_json(output_file, all_results, extra)
    _save_markdown_report(output_file, all_results, extra)
    print_header("Ответы моделей")
    show_model_responses(all_results)
    print_header("Сравнение метрик")
    show_metrics_table(all_results)
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────────


def main():
    """Точка входа: интерактивный режим или разовый запрос."""
    parser = argparse.ArgumentParser(
        description="День 5: Сравнение слабой / средней / сильной модели HuggingFace"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Одиночный запрос (без интерактивного режима)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Устройство: cpu, mps, cuda",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Максимум токенов в ответе (по умолчанию: 256)",
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

    device = args.device or detect_device()

    # Определяем цель вывода: явный файл или авто-сохранение в output/
    output_target = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output"
    )

    # Режим: одиночный запрос (без интерактива)
    if args.prompt:
        print(f"  Устройство: {device}")
        output_file = _resolve_output_file(output_target, args.prompt)
        process_single_query(
            args.prompt, device, args.max_tokens, args.temperature, output_file
        )
        return

    # ── Интерактивный режим ────────────────────────────────────────────────────
    print()
    print("=" * (BOX_WIDTH + 2))
    print("  День 5: Версии моделей — сравнение слабой / средней / сильной")
    print("=" * (BOX_WIDTH + 2))
    print()
    print(f"  Устройство: {device.upper()}")
    print("  Модели:")
    for mc in MODELS:
        print(f"    • {mc['label']} ({mc['params_b']}B) — {mc['name']}")
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
            query, device, args.max_tokens, args.temperature, output_file
        )


if __name__ == "__main__":
    main()
