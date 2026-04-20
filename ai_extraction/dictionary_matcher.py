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


def best_match(extracted: str, dictionary: list[str]) -> tuple[str | None, int]:
    value = (extracted or "").strip()
    if not value or not dictionary:
        return None, 9999

    winner = None
    winner_distance = 10**9
    for entry in dictionary:
        dist = levenshtein_distance(value, entry)
        if dist < winner_distance:
            winner_distance = dist
            winner = entry
    return winner, int(winner_distance)


def compute_C_dict(extracted: str, best: str | None, distance: int) -> float:
    if not extracted or not best:
        return 0.0
    denom = max(len(extracted), len(best), 1)
    return max(0.0, 1.0 - (float(distance) / float(denom)))


# Backwards-compatible alias
closest_dictionary_match = best_match
