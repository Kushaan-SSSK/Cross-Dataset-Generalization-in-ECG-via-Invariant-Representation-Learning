import re
from pathlib import Path
import csv

ROOT = Path("outputs/shortcuts")
runs = sorted([p for p in ROOT.iterdir() if p.is_dir()])

# Common patterns that show up in Hydra/Lightning/custom logs
PATTERNS = [
    ("best_val_acc", re.compile(r"(best.*val.*acc(?:uracy)?)[^\d]*([0-9]*\.[0-9]+|[0-9]+)")),

    ("val_acc",      re.compile(r"(val.*acc(?:uracy)?)[^\d]*([0-9]*\.[0-9]+|[0-9]+)")),
    ("valid_acc",    re.compile(r"(valid.*acc(?:uracy)?)[^\d]*([0-9]*\.[0-9]+|[0-9]+)")),
    ("test_acc",     re.compile(r"(test.*acc(?:uracy)?)[^\d]*([0-9]*\.[0-9]+|[0-9]+)")),

    # If you log "Accuracy: 0.873" or similar
    ("acc",          re.compile(r"(^|\b)acc(?:uracy)?[^\d]*([0-9]*\.[0-9]+|[0-9]+)", re.IGNORECASE)),
]

def extract_metrics(log_text: str):
    matches = []
    for name, pat in PATTERNS:
        for m in pat.finditer(log_text):
            # value is always group 2 for patterns above
            try:
                val = float(m.group(2))
            except Exception:
                continue
            # keep some context
            ctx = m.group(0).strip()
            matches.append((name, val, ctx))
    return matches

rows = []
for run_dir in runs:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        continue

    text = log_path.read_text(errors="ignore")

    metrics = extract_metrics(text)

    # Heuristic: prefer best_val_acc, then val_acc, then test_acc, then acc
    chosen = None
    preference = ["best_val_acc", "val_acc", "valid_acc", "test_acc", "acc"]
    for key in preference:
        cand = [m for m in metrics if m[0] == key]
        if cand:
            # take the LAST occurrence (usually final/best reported late)
            chosen = cand[-1]
            break

    row = {
        "run": run_dir.name,
        "log_path": str(log_path),
        "metric_name": chosen[0] if chosen else "",
        "metric_value": chosen[1] if chosen else "",
        "matched_text": chosen[2] if chosen else "",
    }
    rows.append(row)

out_path = Path("poisoned_results.csv")
with out_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run","log_path","metric_name","metric_value","matched_text"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {out_path} with {len(rows)} runs.")
