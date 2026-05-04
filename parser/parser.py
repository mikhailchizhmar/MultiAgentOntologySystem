"""
parser.py — универсальный парсер PDF и веб-страниц в TXT
=========================================================
Использование:
    python parser.py urls.txt
    python parser.py urls.txt --output ./output --delay 1.5 --short-threshold 80

Формат urls.txt (одна запись на строку):
    https://example.com/doc.pdf  | имя_файла  | тип_документа
    https://example.com/page     | имя_файла  | тип_документа

Специальные типы с кастомным скрапингом:
    centrinvest_mortgage   — парсит блоки ипотечных программ после «Полезная информация»
    centrinvest_deposits   — переходит по каждой «Подробнее о вкладе» и собирает тексты

Структура вывода:
    output/
    ├── parsed_docs/      ← нормальные документы (>= порога строк)
    ├── short_docs/       ← короткие документы + short_docs.log
    └── parsed_docs/parser.log  ← полный лог сессии
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

    # Не доверяем дефолтному latin-1, который requests ставит когда charset не указан
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
# ПАРСИНГ PDF
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
# ПАРСИНГ ОБЫЧНЫХ ВЕБ-СТРАНИЦ
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
# КАСТОМНЫЙ ПАРСИНГ: ЦЕНТР-ИНВЕСТ ИПОТЕКА
# ══════════════════════════════════════════════════════════════════════════════

def parse_centrinvest_mortgage(url: str, logger) -> str:
    """
    Парсит /for-individuals/mortgage.
    Собирает все блоки ПОСЛЕ элемента «Полезная информация»:
    заголовки программ + описания → один текст.
    """
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html, enc = fix_encoding(resp, logger)
    logger.debug(f"Кодировка (mortgage): {enc}")
    soup = BeautifulSoup(html, "html.parser")

    # Ищем маркер «Полезная информация»
    marker = None
    for el in soup.find_all(string=re.compile(r"Полезная информация", re.I)):
        parent = el.find_parent()
        if parent:
            marker = parent
            break

    blocks = []

    if marker:
        for sibling in marker.find_all_next():
            tag = sibling.name
            if not tag:
                continue
            if tag in ("h1", "h2", "h3", "h4", "h5"):
                t = sibling.get_text(strip=True)
                if t:
                    blocks.append(f"\n{'=' * 50}\n{t}\n{'=' * 50}")
            elif tag == "p":
                t = sibling.get_text(separator=" ", strip=True)
                if t and len(t) > 15:
                    blocks.append(t)
            elif tag == "li":
                t = sibling.get_text(separator=" ", strip=True)
                if t and len(t) > 10:
                    blocks.append(f"• {t}")
            elif tag == "div":
                # Только листовые div (без вложенных блоков)
                if not sibling.find(["div", "section", "article", "ul", "ol"]):
                    t = sibling.get_text(separator=" ", strip=True)
                    if t and len(t) > 20:
                        blocks.append(t)
    else:
        logger.warning("Маркер 'Полезная информация' не найден — парсим секции страницы")
        for sec in soup.find_all(["section", "article"],
                                 class_=re.compile(r"(product|credit|card|block|program)", re.I)):
            t = sec.get_text(separator="\n", strip=True)
            if len(t) > 50:
                blocks.append(t)

    if not blocks:
        logger.warning("Блоков не найдено — возврат полного текста страницы")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    return "\n\n".join(blocks)


# ══════════════════════════════════════════════════════════════════════════════
# КАСТОМНЫЙ ПАРСИНГ: ЦЕНТР-ИНВЕСТ ВКЛАДЫ
# ══════════════════════════════════════════════════════════════════════════════

def parse_centrinvest_deposits(base_url: str, logger) -> str:
    """
    1. Загружает /for-individuals/deposits.
    2. Собирает все ссылки «Подробнее о вкладе» / «Подробнее».
    3. Переходит по каждой ссылке, парсит страницу вклада.
    4. Склеивает всё в один текст с разделителями.
    """
    resp = requests.get(base_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html, enc = fix_encoding(resp, logger)
    logger.debug(f"Кодировка (deposits-index): {enc}")
    soup = BeautifulSoup(html, "html.parser")

    # Извлекаем домен для построения абсолютных ссылок
    domain_match = re.match(r"(https?://[^/]+)", base_url)
    domain = domain_match.group(1) if domain_match else ""

    deposit_links = []
    seen = set()
    link_re = re.compile(r"подробнее", re.I)

    for a in soup.find_all("a", href=True):
        link_text = a.get_text(strip=True)
        href = a["href"].strip()
        if not link_re.search(link_text):
            continue
        if href.startswith("http"):
            full = href
        elif href.startswith("/"):
            full = domain + href
        else:
            full = base_url.rstrip("/") + "/" + href
        if full not in seen:
            seen.add(full)
            deposit_links.append(full)
            logger.debug(f"Ссылка на вклад: {full}")

    if not deposit_links:
        logger.warning("Ссылки 'Подробнее' не найдены — парсим саму страницу вкладов")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    logger.info(f"Найдено {len(deposit_links)} карточек вкладов")
    all_blocks = []

    for i, dep_url in enumerate(deposit_links, 1):
        logger.info(f"  [{i}/{len(deposit_links)}] {dep_url}")
        try:
            dep_resp = requests.get(dep_url, headers=HEADERS, timeout=30)
            dep_resp.raise_for_status()
            dep_html, dep_enc = fix_encoding(dep_resp, logger)

            dep_text = trafilatura.extract(
                dep_html,
                include_tables=True,
                include_links=False,
                favor_recall=True,
            )
            if not dep_text or len(dep_text.splitlines()) < 5:
                dep_soup = BeautifulSoup(dep_html, "html.parser")
                for tag in dep_soup(["script", "style", "nav", "footer",
                                     "header", "aside", "noscript"]):
                    tag.decompose()
                dep_text = dep_soup.get_text(separator="\n")

            dep_text = clean_text(dep_text)
            if dep_text:
                separator = "═" * 60
                all_blocks.append(
                    f"{separator}\nВКЛАД: {dep_url}\n{separator}\n\n{dep_text}"
                )
            time.sleep(0.8)
        except Exception as e:
            logger.error(f"Ошибка при загрузке {dep_url}: {e}")

    return "\n\n".join(all_blocks)


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
        if doc_type == "centrinvest_mortgage":
            raw = parse_centrinvest_mortgage(url, logger)

        elif doc_type == "centrinvest_deposits":
            raw = parse_centrinvest_deposits(url, logger)

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
    ap.add_argument("--output", default="./parsed_docs",
                    help="Папка для нормальных документов (по умолчанию: ./parsed_docs)")
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
        if i < total:
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
