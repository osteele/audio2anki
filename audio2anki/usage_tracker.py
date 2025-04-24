import contextvars
import threading
from collections import defaultdict
from typing import TypedDict

from rich.console import Console
from rich.table import Table


class UsageTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.usage_data: dict[str, dict[str, int | float]] = defaultdict(
            lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
                "character_cost": 0,
                "minutes": 0.0,
            }
        )

    def record_usage(
        self,
        *,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        character_cost: int | None = None,
        minutes: float | None = None,
    ) -> None:
        with self.lock:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
            self.usage_data[model]["input_tokens"] += input_tokens or 0
            self.usage_data[model]["output_tokens"] += output_tokens or 0
            self.usage_data[model]["character_cost"] += character_cost or 0
            self.usage_data[model]["minutes"] += minutes or 0.0
            self.usage_data[model]["total_tokens"] += total_tokens
            self.usage_data[model]["calls"] += 1

    def get_summary(self) -> dict[str, dict[str, int | float]]:
        with self.lock:
            return dict(self.usage_data)

    def report(self) -> None:
        summary = self.get_summary()
        if not summary:
            return
        for model, data in summary.items():
            print(f"Model: {model}")
            print(f"  Calls: {data['calls']}")
            print(f"  Input Tokens: {data['input_tokens']}")
            print(f"  Output Tokens: {data['output_tokens']}")
            print(f"  Total Tokens: {data['total_tokens']}")
            print(f"  Minutes: {data['minutes']:.2f}")
            print(f"  Character Cost: {data['character_cost']}")

    def render_usage_table(self, console: Console) -> None:
        summary = self.get_summary()
        if not summary:
            return
        # Determine which columns have any nonzero values
        columns = [
            ("Calls", "calls"),
            ("Input Tokens", "input_tokens"),
            ("Output Tokens", "output_tokens"),
            ("Character Cost", "character_cost"),
            ("Minutes", "minutes"),
            ("Total Tokens", "total_tokens"),
        ]
        nonzero_columns: list[tuple[str, str]] = []
        for title, key in columns:
            if any(data.get(key, 0) for data in summary.values()):
                nonzero_columns.append((title, key))
        if not nonzero_columns:
            return
        table = Table(title="OpenAI Usage Report")
        table.add_column("Model", style="bold")
        for title, _ in nonzero_columns:
            table.add_column(title, justify="right")
        for model, data in summary.items():
            row = [model]
            for _, key in nonzero_columns:
                val = data.get(key, 0)
                val_str = f"{val:.2f}" if key == "minutes" else str(val)
                row.append(val_str if val else "")
            table.add_row(*row)
        console.print(table)


current_tracker: contextvars.ContextVar["UsageTracker | None"] = contextvars.ContextVar("current_tracker", default=None)


class ApiUsage(TypedDict, total=False):
    input_tokens: int | None
    output_tokens: int | None
    character_cost: int | None
    minutes: float | None


def record_api_usage(
    model: str,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    character_cost: int | None = None,
    minutes: float | None = None,
) -> None:
    tracker = current_tracker.get()
    if tracker:
        tracker.record_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            character_cost=character_cost,
            minutes=minutes,
        )
