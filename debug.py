from ocr_positions import normalize_text, NORMALIZED_NAMES
import difflib
line="Semiconductor ETF"
norm=normalize_text(line)
print(norm)
print(difflib.get_close_matches(norm, list(NORMALIZED_NAMES.keys()), n=1, cutoff=0.6))

