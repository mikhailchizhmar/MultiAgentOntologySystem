"""
corpus/loader.py
─────────────────
Загружает документы из parser/parsed_docs и gold_annotations.json.

Экспортирует:
  load_corpus(doc_ids)  → list[dict]  все документы для пайплайна
  load_gold()           → dict        gold_annotations для evaluate()
  PARSED_DOCS_DIR       Path до папки с .txt файлами
"""

from __future__ import annotations

import re
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ПУТИ
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).parent.parent
PARSED_DOCS_DIR = PROJECT_ROOT / "parser" / "parsed_docs"
GOLD_PATH       = PROJECT_ROOT / "corpus" / "gold_annotations.json"
URLS_PATH       = PROJECT_ROOT / "parser" / "urls.txt"

# ─────────────────────────────────────────────────────────────────────────────
# МАППИНГ ТИПА ИЗ urls.txt → короткий тип
# ─────────────────────────────────────────────────────────────────────────────

TYPE_MAP = {
    "кредитный_договор":    "credit",
    "депозитный_продукт":   "deposit",
    "нормативный_документ": "regulatory",
    "инвестиционный":       "investment",
    "страховой":            "insurance",
}


# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def _load_url_types() -> dict[str, str]:
    """Читает urls.txt → {doc_id: short_type}."""
    result: dict[str, str] = {}
    if not URLS_PATH.exists():
        return result
    with open(URLS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            result[parts[1]] = TYPE_MAP.get(parts[2], parts[2])
    return result


def _read_parsed_doc(filename: str) -> str:
    """
    Читает .txt файл из parsed_docs.
    Файлы имеют заголовок [SOURCE]/[PATH]/[TYPE] и горизонтальную черту-разделитель.
    Возвращает текст после черты.
    """
    path = PARSED_DOCS_DIR / filename
    if not path.exists():
        return ""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    sep = re.search(r'[-─]{10,}', raw)
    if sep:
        return raw[sep.end():].strip()
    lines = raw.splitlines()
    return "\n".join(
        l for l in lines
        if not l.startswith(("[SOURCE]", "[PATH]", "[TYPE]"))
    ).strip()


# ─────────────────────────────────────────────────────────────────────────────
# ПУБЛИЧНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def load_gold() -> dict:
    """
    Загружает gold_annotations.json.
    Дополняет каждый документ:
      - type  из urls.txt (если в gold = None)
      - text  из parser/parsed_docs (если в gold пустой)
    """
    with open(GOLD_PATH, encoding="utf-8") as f:
        data = json.load(f)

    url_types = _load_url_types()

    for doc in data["documents"]:
        if not doc.get("type"):
            doc["type"] = url_types.get(doc["id"], "unknown")
        if not doc.get("text") and doc.get("source"):
            doc["text"] = _read_parsed_doc(doc["source"])

    return data


def load_corpus(doc_ids: list[str] | None = None) -> list[dict]:
    """
    Загружает все документы из parser/parsed_docs.

    Сначала идут документы с gold-разметкой, затем остальные.
    doc_ids — опциональный фильтр по id.

    Возвращает list[dict] с полями: id, type, product, source, text.
    """
    url_types = _load_url_types()
    gold_data = load_gold()
    gold_ids  = {doc["id"] for doc in gold_data["documents"]}

    docs: list[dict] = []

    # Документы с разметкой — из gold (text и type уже дополнены load_gold)
    for doc in gold_data["documents"]:
        docs.append({
            "id":      doc["id"],
            "type":    doc.get("type", "unknown"),
            "product": doc.get("title", doc["id"]),
            "source":  doc.get("source", ""),
            "text":    doc.get("text", ""),
        })

    # Остальные документы из parsed_docs (без разметки, для inference)
    for txt_path in sorted(PARSED_DOCS_DIR.glob("*.txt")):
        if txt_path.name == "parser.log":
            continue
        doc_id = txt_path.stem
        if doc_id in gold_ids:
            continue
        text = _read_parsed_doc(txt_path.name)
        if not text:
            continue
        docs.append({
            "id":      doc_id,
            "type":    url_types.get(doc_id, "unknown"),
            "product": doc_id.replace("_", " "),
            "source":  txt_path.name,
            "text":    text,
        })

    if doc_ids:
        docs = [d for d in docs if d["id"] in doc_ids]

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# БЫСТРАЯ ПРОВЕРКА
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gold = load_gold()
    print(f"Gold: {len(gold['documents'])} документов")
    for doc in gold["documents"]:
        print(f"  [{doc['id']:<35}] type={doc['type']:<12} "
              f"text_len={len(doc.get('text', ''))}")

    corpus = load_corpus()
    print(f"\nCorpus: {len(corpus)} документов")
    for doc in corpus:
        print(f"  [{doc['id']:<35}] type={doc['type']:<12} "
              f"text_len={len(doc['text'])}")
