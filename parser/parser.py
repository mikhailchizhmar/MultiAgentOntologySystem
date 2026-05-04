"""
parser.py — универсальный парсер PDF и веб-страниц в TXT
Использование:
    python parser.py urls.txt
    python parser.py urls.txt --output ./output --delay 1.5

Формат urls.txt (одна запись на строку):
    https://example.com/doc.pdf | кредитный_договор | кредитный_договор
    https://example.com/page    | депозит_альфа     | депозитный_продукт
    ^URL                          ^имя файла          ^тип документа
"""

import sys
import re
import time
import argparse
from pathlib import Path
import trafilatura
import pdfplumber
import requests

# ── настройки ─────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


# ── вспомогательные функции ───────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Убирает лишние пробелы и пустые строки."""
    lines = text.splitlines()
    cleaned = []
    prev_blank = False
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


def safe_filename(name: str) -> str:
    """Превращает произвольную строку в безопасное имя файла."""
    name = re.sub(r"[^\w\-]", "_", name, flags=re.UNICODE)
    return name[:80]


def make_header(source: str, doc_type: str) -> str:
    return (
        f"[SOURCE] {source}\n"
        f"[TYPE] {doc_type}\n"
        f"{'─' * 60}\n\n"
    )


# ── парсинг PDF ───────────────────────────────────────────────────────────────

def parse_pdf_from_url(url: str) -> str:
    """Скачивает PDF и извлекает текст через pdfplumber."""
    import io
    response = requests.get(url, headers=HEADERS, timeout=30, stream=True)
    response.raise_for_status()

    pdf_bytes = response.content
    text_parts = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n\n".join(text_parts)


# ── парсинг веб-страниц ───────────────────────────────────────────────────────

def parse_web_page(url: str) -> str:
    """Скачивает HTML и извлекает основной текст через trafilatura."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    # trafilatura сам отсекает навигацию, рекламу, футеры
    text = trafilatura.extract(
        response.text,
        include_tables=True,
        include_links=False,
        include_images=False,
        no_fallback=False,
        favor_recall=True,   # предпочитать полноту текста
    )

    if not text:
        # резервный вариант — грубая очистка через BeautifulSoup если trafilatura вернул пусто
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        except ImportError:
            text = ""

    return text or ""


# ── основной процессор ────────────────────────────────────────────────────────

def process_entry(url: str, name: str, doc_type: str, output_dir: Path) -> bool:
    """Парсит один URL и сохраняет результат."""
    url = url.strip()
    name = safe_filename(name.strip())
    doc_type = doc_type.strip()

    out_path = output_dir / f"{name}.txt"

    print(f"  → {url[:70]}...")

    try:
        # определяем тип по URL или Content-Type
        is_pdf = url.lower().endswith(".pdf")

        if not is_pdf:
            # делаем HEAD-запрос чтобы проверить Content-Type
            try:
                head = requests.head(url, headers=HEADERS, timeout=10, allow_redirects=True)
                ct = head.headers.get("Content-Type", "")
                is_pdf = "pdf" in ct.lower()
            except Exception:
                pass

        if is_pdf:
            raw_text = parse_pdf_from_url(url)
            source_label = url
        else:
            raw_text = parse_web_page(url)
            source_label = url

        text = clean_text(raw_text)

        if not text:
            print(f"  ⚠️  Пустой результат для {name} — пропускаем")
            return False

        header = make_header(source_label, doc_type)
        full_content = header + text

        out_path.write_text(full_content, encoding="utf-8")
        size_kb = len(full_content.encode("utf-8")) / 1024
        print(f"  ✅  {name}.txt  ({size_kb:.1f} KB, {len(text.splitlines())} строк)")
        return True

    except requests.exceptions.HTTPError as e:
        print(f"  ❌  HTTP {e.response.status_code} для {url}")
    except requests.exceptions.ConnectionError:
        print(f"  ❌  Нет соединения: {url}")
    except requests.exceptions.Timeout:
        print(f"  ❌  Таймаут: {url}")
    except Exception as e:
        print(f"  ❌  Ошибка [{type(e).__name__}]: {e}")

    return False


# ── чтение файла со списком URL ───────────────────────────────────────────────

def read_urls_file(path: str) -> list[tuple[str, str, str]]:
    """
    Читает файл со списком документов.
    Формат строки: URL | имя_файла | тип_документа
    Строки начинающиеся с # — комментарии, пропускаются.
    """
    entries = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                print(f"  ⚠️  Строка {lineno}: неверный формат, пропускаем → {line}")
                continue
            url, name, doc_type = parts[0], parts[1], parts[2]
            entries.append((url, name, doc_type))
    return entries


# ── точка входа ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Парсер PDF и веб-страниц → TXT с метаданными"
    )
    parser.add_argument("urls_file", help="Путь к файлу со списком URL (urls.txt)")
    parser.add_argument(
        "--output", default="./parsed_docs",
        help="Папка для результатов (по умолчанию: ./parsed_docs)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Задержка между запросами в секундах (по умолчанию: 1.0)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = read_urls_file(args.urls_file)
    total = len(entries)

    print(f"\n📂  Папка вывода: {output_dir.resolve()}")
    print(f"📋  Документов к обработке: {total}")
    print(f"⏱️   Задержка между запросами: {args.delay}s\n")
    print("─" * 60)

    success, failed = 0, 0

    for i, (url, name, doc_type) in enumerate(entries, start=1):
        print(f"[{i}/{total}] {name} ({doc_type})")
        ok = process_entry(url, name, doc_type, output_dir)
        if ok:
            success += 1
        else:
            failed += 1

        if i < total:
            time.sleep(args.delay)

    print("\n" + "─" * 60)
    print(f"✅  Успешно: {success}   ❌  Ошибки: {failed}   📄  Всего: {total}")
    print(f"📂  Результаты сохранены в: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
