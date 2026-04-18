import csv
from pathlib import Path

from .confidence import levenshtein_distance


class DictionaryStore:
    def __init__(self, dictionaries_dir: str | Path):
        self.path = Path(dictionaries_dir)
        self._data: dict[str, list[str]] = {}

    def load(self) -> dict[str, list[str]]:
        self._data = {}
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            return self._data

        for csv_path in sorted(self.path.glob("*.csv")):
            names: list[str] = []
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                if "name" not in (reader.fieldnames or []):
                    continue
                for row in reader:
                    value = (row.get("name") or "").strip()
                    if value:
                        names.append(value)
            self._data[csv_path.name] = names
        return self._data

    def get(self, filename: str) -> list[str]:
        return self._data.get(filename, [])


def closest_dictionary_match(value: str, entries: list[str]) -> tuple[str | None, int]:
    value = (value or "").strip()
    if not value or not entries:
        return None, 9999

    best = None
    best_dist = 10**9
    for candidate in entries:
        d = levenshtein_distance(value, candidate)
        if d < best_dist:
            best_dist = d
            best = candidate
    return best, int(best_dist)
