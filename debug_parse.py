from pathlib import Path
from ocr_positions import normalize_text, NORMALIZED_NAMES, DISPLAY_TO_TICKER
import difflib
lines=[ln.strip() for ln in Path("ocr_debug.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
for idx,line in enumerate(lines):
    norm=normalize_text(line)
    match=difflib.get_close_matches(norm, list(NORMALIZED_NAMES.keys()), n=1, cutoff=0.5)
    if match:
        print(idx,line,match[0])

