"""CLI для запуска пайплайна обработки записей тьютора.

Примеры:
    python main.py
    python main.py run --mode fallback
    python main.py sweep
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from tutor_assistant.pipeline import load_dataset, run_pipeline, sweep_unknown_margin
from tutor_assistant.report import render_markdown, sweep_table

DEFAULT_DATASET = Path("dataset.json")
DEFAULT_VALIDATION = Path("data/classification_validation.json")
DEFAULT_OUTPUT_DIR = Path("output")
MODES: dict[str, bool | None] = {"auto": None, "llm": True, "fallback": False}


def build_parser() -> argparse.ArgumentParser:
    """Описывает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="AI-помощник тьютора: обработка записей")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="прогнать пайплайн на датасете (по умолчанию)")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument(
        "--mode",
        choices=MODES,
        default="auto",
        help="auto - GigaChat при наличии ключа, llm - только GigaChat, fallback - ключевые слова",
    )

    sweep = subparsers.add_parser("sweep", help="перебрать порог unknown на валидационной выборке")
    sweep.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    return parser


def run_command(dataset_path: Path, output_dir: Path, mode: str) -> None:
    """Запускает пайплайн и сохраняет report.json и tables.md."""
    report = run_pipeline(load_dataset(dataset_path), MODES[mode])
    report["summary"]["mode"] = mode

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"report_{mode}.json"
    tables_path = output_dir / f"tables_{mode}.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    tables_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"JSON-отчёт: {report_path}\nТаблицы: {tables_path}")


def sweep_command(validation_path: Path) -> None:
    """Печатает таблицу точности и доли unknown для разных порогов."""
    samples = json.loads(validation_path.read_text(encoding="utf-8"))
    print(sweep_table(sweep_unknown_margin(samples)))


def main() -> None:
    """Точка входа CLI."""
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()

    if args.command == "sweep":
        sweep_command(args.validation)
    elif args.command == "run":
        run_command(args.dataset, args.output_dir, args.mode)
    else:
        run_command(DEFAULT_DATASET, DEFAULT_OUTPUT_DIR, "auto")


if __name__ == "__main__":
    main()
