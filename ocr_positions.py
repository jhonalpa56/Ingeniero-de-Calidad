import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import difflib

import cv2
import numpy as np
import pytesseract

DEFAULT_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if DEFAULT_TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(DEFAULT_TESSERACT_PATH)


@dataclass
class PositionRecord:
    ticker: str
    name: str
    volume: float
    avg_price: Optional[float]
    market_value: float
    net_pl: float


DISPLAY_TO_TICKER = {
    "Semiconductor": "SMH.L",
    "Innovation": "ARKK",
    "Physical Gold": "IGLN.L",
    "Diversified Commodity Swap": "ICOM.L",
    "Global Clean Energy": "INRG.L",
    "EQQQ Nasdaq-100": "EQQQ.L",
    "Core MSCI World": "SWDA.L",
    "MSCI World SRI": "2B7K.DE",
    "Uranium Miners": "U3O8.DE",
    "Bitcoin": "VBTC.DE",
}

def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


NORMALIZED_NAMES = {normalize_text(name): name for name in DISPLAY_TO_TICKER.keys()}


def preprocess_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def run_ocr(image: np.ndarray) -> str:
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(image, lang="spa+eng", config=config)
    return text


def normalize_number(raw: str) -> float:
    cleaned = raw
    for token in ["USD", "US$", "EUR", "€", "%"]:
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace(",", ".")
    cleaned = re.sub(r"[^0-9\.-]", "", cleaned)
    if not cleaned:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_positions(text: str) -> Dict[str, PositionRecord]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    results: Dict[str, PositionRecord] = {}

    idx = 0
    while idx < len(lines):
        normalized = normalize_text(lines[idx])
        match = difflib.get_close_matches(normalized, NORMALIZED_NAMES.keys(), n=1, cutoff=0.6)
        if match:
            display_name = NORMALIZED_NAMES[match[0]]
            ticker = DISPLAY_TO_TICKER.get(display_name)
            if not ticker:
                idx += 1
                continue
            volume = avg_price = market_value = net_pl = None

            for j in range(1, 8):
                if idx + j >= len(lines):
                    break
                line = lines[idx + j]
                numbers = re.findall(r"-?\d[\d\s\.,]*", line)
                if not numbers:
                    continue
                if volume is None and len(numbers) >= 3:
                    volume = normalize_number(numbers[0])
                    market_value = normalize_number(numbers[1])
                    net_pl = normalize_number(numbers[2])
                    continue
                if avg_price is None and "@" in line:
                    avg_price = normalize_number(numbers[0])

            if volume is not None and market_value is not None:
                record = PositionRecord(
                    ticker=ticker,
                    name=display_name,
                    volume=volume,
                    avg_price=avg_price if avg_price else (market_value / volume if volume else None),
                    market_value=market_value,
                    net_pl=net_pl if net_pl is not None else 0.0,
                )
                results[ticker] = record
            idx += 5
        else:
            idx += 1
    return results


def positions_from_image(path: str) -> Dict[str, Dict[str, float]]:
    image = preprocess_image(Path(path))
    raw_text = run_ocr(image)
    records = parse_positions(raw_text)
    return {
        rec.ticker: {
            "qty": rec.volume,
            "avg": rec.avg_price or 0.0,
            "market_value": rec.market_value,
            "net_pl": rec.net_pl,
        }
        for rec in records.values()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extrae posiciones desde un screenshot de trading")
    parser.add_argument("image_path", help="Ruta al archivo de imagen (png/jpg)")
    parser.add_argument("--output", help="Archivo opcional para guardar JSON")
    parser.add_argument("--debug-text", action="store_true", help="Guarda el texto OCR bruto en ocr_debug.txt")
    args = parser.parse_args()

    image = preprocess_image(Path(args.image_path))
    raw_text = run_ocr(image)
    if args.debug_text:
        Path("ocr_debug.txt").write_text(raw_text, encoding="utf-8")
        print("Texto OCR guardado en ocr_debug.txt")
    positions = parse_positions(raw_text)
    positions = {
        rec.ticker: {
            "qty": rec.volume,
            "avg": rec.avg_price or 0.0,
            "market_value": rec.market_value,
            "net_pl": rec.net_pl,
        }
        for rec in positions.values()
    }

    if args.output:
        Path(args.output).write_text(json.dumps(positions, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Guardado en {args.output}")
    else:
        print(json.dumps(positions, indent=2, ensure_ascii=False))
