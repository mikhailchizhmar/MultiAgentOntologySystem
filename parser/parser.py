"""
parser.py — универсальный парсер PDF и веб-страниц в TXT
=========================================================
Запуск из корня проекта:
    python parser/parser.py parser/urls.txt
    python parser/parser.py parser/urls.txt --output ./parser/parsed_docs --delay 1.5

Формат urls.txt (одна запись на строку):
    https://example.com/doc.pdf   | имя_файла | тип_документа
    https://example.com/page      | имя_файла | тип_документа
    parser/local/file.pdf         | имя_файла | тип_документа   ← локальный PDF

Структура вывода (относительно --output):
    parsed_docs/      ← нормальные документы (>= порога строк)
    short_docs/       ← короткие документы + short_docs.log
    parsed_docs/parser.log
"""

import sys
import re
import io
import time
import logging
import argparse
import unicodedata
from pathlib import Path
from datetime import datetime

# ── проверка зависимостей ─────────────────────────────────────────────────────
MISSING = []
for pkg in ["pdfplumber", "trafilatura", "requests", "bs4", "chardet"]:
    try:
        __import__(pkg)
    except ImportError:
        MISSING.append(pkg if pkg != "bs4" else "beautifulsoup4")

if MISSING:
    print("Не хватает библиотек. Установите командой:")
    print(f"    pip install {' '.join(MISSING)}")
    sys.exit(1)

import pdfplumber
import trafilatura
import requests
import chardet
from bs4 import BeautifulSoup

# ── константы ─────────────────────────────────────────────────────────────────
SHORT_THRESHOLD_DEFAULT = 50

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "parser.log"
    logger = logging.getLogger("parser")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-]", "_", name, flags=re.UNICODE)
    return name[:80]


def make_header(source: str, path: Path, doc_type: str) -> str:
    return f"[SOURCE] {source}\n[PATH] {path}\n[TYPE] {doc_type}\n{'─' * 60}\n\n"


def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned, prev_blank = [], False
    for line in lines:
        line = line.rstrip()
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).strip()


# ══════════════════════════════════════════════════════════════════════════════
# ИСПРАВЛЕНИЕ КОДИРОВКИ
# ══════════════════════════════════════════════════════════════════════════════

def fix_encoding(response: requests.Response, logger) -> tuple[str, str]:
    """
    Надёжное декодирование:
    1. Проверяем объявленную кодировку в заголовке.
    2. Если это дефолтный latin-1/iso-8859-1 — не доверяем, берём chardet.
    3. Пробуем кандидатов по порядку; выбираем тот, где < 1% мусорных символов.
    4. Нормализуем Unicode NFC.
    """
    raw = response.content
    declared = response.encoding or ""

    detected = chardet.detect(raw)
    detected_enc = detected.get("encoding") or "utf-8"

    bad_defaults = {"iso-8859-1", "latin-1", "windows-1252", "ascii"}
    candidates = []
    if declared and declared.lower() not in bad_defaults:
        candidates.append(declared)
    candidates.append(detected_enc)
    candidates += ["utf-8", "cp1251", "latin-1"]

    for enc in candidates:
        try:
            decoded = raw.decode(enc)
            ratio = decoded.count("\ufffd") / max(len(decoded), 1)
            if ratio < 0.01:
                return unicodedata.normalize("NFC", decoded), enc
        except (UnicodeDecodeError, LookupError):
            continue

    fallback = raw.decode("utf-8", errors="replace")
    return unicodedata.normalize("NFC", fallback), "utf-8(forced)"


# ══════════════════════════════════════════════════════════════════════════════
# ПАРСИНГ УДАЛЁННЫХ PDF
# ══════════════════════════════════════════════════════════════════════════════

def parse_pdf(url: str, logger) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=45, stream=True)
    resp.raise_for_status()
    parts = []
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    if not parts:
        logger.warning("PDF без текстового слоя (скан?) — текст не извлечён")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# ПАРСИНГ ЛОКАЛЬНЫХ PDF
# ══════════════════════════════════════════════════════════════════════════════

def parse_local_pdf(path: str, logger) -> str:
    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"Локальный файл не найден: {local_path.resolve()}")
    parts = []
    with pdfplumber.open(local_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    if not parts:
        logger.warning("PDF без текстового слоя (скан?) — текст не извлечён")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# ПАРСИНГ ВЕБ-СТРАНИЦ
# ══════════════════════════════════════════════════════════════════════════════

def parse_web(url: str, logger) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html, enc = fix_encoding(resp, logger)
    logger.debug(f"Кодировка: {enc}")

    text = trafilatura.extract(
        html,
        include_tables=True,
        include_links=False,
        include_images=False,
        no_fallback=False,
        favor_recall=True,
    )

    if not text or len(text.splitlines()) < 5:
        logger.debug("trafilatura — мало текста, резерв BeautifulSoup")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n")

    return text or ""


# ══════════════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ ПРОЦЕССОР
# ══════════════════════════════════════════════════════════════════════════════

def process_entry(
    url, name, doc_type,
    output_dir, short_dir,
    short_threshold, short_log_path,
    logger,
) -> tuple[bool, str]:

    url = url.strip()
    name = safe_filename(name.strip())
    doc_type = doc_type.strip()
    logger.info(f">>> {name}  [{doc_type}]  {url[:80]}")

    try:
        # Выбор стратегии
        if url.startswith("parser/local/"):
            raw = parse_local_pdf(url, logger)

        else:
            is_pdf = url.lower().endswith(".pdf")
            if not is_pdf:
                try:
                    head = requests.head(url, headers=HEADERS,
                                        timeout=10, allow_redirects=True)
                    ct = head.headers.get("Content-Type", "")
                    is_pdf = "pdf" in ct.lower()
                except Exception:
                    pass
            raw = parse_pdf(url, logger) if is_pdf else parse_web(url, logger)

        text = clean_text(raw)

        if not text:
            logger.warning("Пустой результат — пропускаем")
            return False, "empty"

        line_count = len(text.splitlines())
        full_content = make_header(url, output_dir / f"{name}.txt", doc_type) + text
        size_kb = len(full_content.encode("utf-8")) / 1024

        if line_count < short_threshold:
            out_path = short_dir / f"{name}.txt"
            out_path.write_text(full_content, encoding="utf-8")
            with open(short_log_path, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%H:%M:%S")
                f.write(f"{ts}  {name}.txt  |  {line_count} строк  |  {size_kb:.1f} KB  |  {url}\n")
            logger.warning(
                f"КОРОТКИЙ ({line_count} стр. < {short_threshold}) → short_docs/{name}.txt"
            )
            return True, "short"
        else:
            out_path = output_dir / f"{name}.txt"
            out_path.write_text(full_content, encoding="utf-8")
            logger.info(f"OK  {name}.txt  ({size_kb:.1f} KB, {line_count} строк)")
            return True, "ok"

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP {e.response.status_code}: {url}")
    except requests.exceptions.ConnectionError:
        logger.error(f"Нет соединения: {url}")
    except requests.exceptions.Timeout:
        logger.error(f"Таймаут: {url}")
    except Exception as e:
        logger.error(f"[{type(e).__name__}] {e}")

    return False, "error"


# ══════════════════════════════════════════════════════════════════════════════
# ЧТЕНИЕ urls.txt
# ══════════════════════════════════════════════════════════════════════════════

def read_urls_file(path: str, logger) -> list:
    entries = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                logger.warning(f"Строка {lineno}: неверный формат → {line}")
                continue
            entries.append((parts[0], parts[1], parts[2]))
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Парсер PDF и веб-страниц → TXT")
    ap.add_argument("urls_file", help="Путь к файлу со списком URL")
    ap.add_argument("--output", default="./parser/parsed_docs",
                    help="Папка для нормальных документов (по умолчанию: ./parser/parsed_docs)")
    ap.add_argument("--delay", type=float, default=1.2,
                    help="Задержка между запросами, сек (по умолчанию: 1.2)")
    ap.add_argument("--short-threshold", type=int, default=SHORT_THRESHOLD_DEFAULT,
                    help=f"Порог 'короткого' документа в строках (по умолчанию: {SHORT_THRESHOLD_DEFAULT})")
    args = ap.parse_args()

    output_dir = Path(args.output)
    short_dir  = output_dir.parent / "short_docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    short_dir.mkdir(parents=True, exist_ok=True)

    short_log_path = short_dir / "short_docs.log"
    if not short_log_path.exists():
        with open(short_log_path, "w", encoding="utf-8") as f:
            f.write(f"# Документы короче порога ({args.short_threshold} строк)\n")
            f.write(f"# Сессия: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("─" * 80 + "\n")

    logger = setup_logging(output_dir)
    logger.info(f"Старт | порог коротких: {args.short_threshold} строк | задержка: {args.delay}s")

    entries = read_urls_file(args.urls_file, logger)
    total = len(entries)
    logger.info(f"Документов: {total}")
    print("─" * 60)

    stats = {"ok": 0, "short": 0, "empty": 0, "error": 0}

    for i, (url, name, doc_type) in enumerate(entries, start=1):
        print(f"\n[{i}/{total}]")
        _, status = process_entry(
            url, name, doc_type,
            output_dir, short_dir,
            args.short_threshold, short_log_path,
            logger,
        )
        stats[status] = stats.get(status, 0) + 1
        if i < total and not url.startswith("parser/local/"):
            time.sleep(args.delay)

    print("\n" + "═" * 60)
    logger.info(
        f"ИТОГО: OK={stats['ok']}  КОРОТКИХ={stats['short']}  "
        f"ПУСТО={stats['empty']}  ОШИБОК={stats['error']}  ВСЕГО={total}"
    )
    logger.info(f"Нормальные  → {output_dir.resolve()}")
    logger.info(f"Короткие    → {short_dir.resolve()}")
    logger.info(f"Лог коротких→ {short_log_path.resolve()}")


if __name__ == "__main__":
    main()
