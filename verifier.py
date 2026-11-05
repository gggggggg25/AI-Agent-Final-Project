"""Flag numbers in the report that don't appear in results.json."""
import re


def _collect_numbers(obj, acc):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, acc)


def check_numbers(report, results):
    known = []
    _collect_numbers(results, known)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", report)  # ignore image refs
    suspicious = []
    for m in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?", text):
        raw = m.group()
        val = float(raw)
        if "." not in raw and abs(val) <= 20:  # small integers: grades, counts, list numbers
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(val - k) <= tol or abs(val - k * 100) <= tol for k in known):
            suspicious.append(raw)
    return sorted(set(suspicious))
