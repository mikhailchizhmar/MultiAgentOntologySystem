"""
review_annotations.py
──────────────────────
Инструмент для ручной проверки автоматической разметки.

Показывает рядом:
  - текст документа
  - сущности и отношения от LLM
  - (если есть) эталонную разметку из gold_standard.json

Использование:
    python review_annotations.py                  # все размеченные документы
    python review_annotations.py --doc doc_001    # конкретный документ
    python review_annotations.py --compare        # только те, для которых есть gold
"""

import json
import argparse
from pathlib import Path

ANNOTATIONS_DIR = Path(__file__).parent / "annotations"
GOLD_PATH       = Path(__file__).parent / "gold_standard.json"

# ANSI-цвета для терминала
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"

CLASS_COLORS = {
    "FinancialProduct": C.GREEN,
    "ProductAttribute": C.BLUE,
    "Actor":            C.YELLOW,
    "Process":          C.CYAN,
    "Condition":        C.RED,
    "LegalTerm":        C.GRAY,
    "Metric":           "\033[95m",  # magenta
}


def load_gold():
    if not GOLD_PATH.exists():
        return {}
    with open(GOLD_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {doc["id"]: doc for doc in data.get("documents", [])}


def load_annotation(doc_id: str) -> dict | None:
    path = ANNOTATIONS_DIR / f"{doc_id}_annotation.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def color_entity(text: str, cls: str) -> str:
    c = CLASS_COLORS.get(cls, "")
    return f"{c}{C.BOLD}{text}{C.RESET}{C.GRAY}[{cls}]{C.RESET}"


def print_header(title: str):
    print(f"\n{C.BOLD}{'─'*60}{C.RESET}")
    print(f"{C.BOLD}  {title}{C.RESET}")
    print(f"{C.BOLD}{'─'*60}{C.RESET}")


def print_entities(entities: list, label: str):
    print(f"\n  {C.BOLD}{label} ({len(entities)} сущностей):{C.RESET}")
    for e in entities:
        conf  = e.get("confidence", "?")
        cls   = e.get("class", "?")
        text  = e.get("text", "?")
        color = CLASS_COLORS.get(cls, "")
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
        print(f"    {color}▸ {text:<35}{C.RESET} "
              f"{C.GRAY}{cls:<20}{C.RESET} conf={conf_str}")
        if e.get("comment"):
            print(f"      {C.GRAY}  ↳ {e['comment']}{C.RESET}")


def print_relations(relations: list, entities_map: dict, label: str):
    print(f"\n  {C.BOLD}{label} ({len(relations)} отношений):{C.RESET}")
    for r in relations:
        subj_id  = r.get("subject", "?")
        obj_id   = r.get("object", "?")
        rel      = r.get("relation", "?")
        conf     = r.get("confidence", "?")
        evidence = r.get("evidence", "")

        subj_text = entities_map.get(subj_id, {}).get("text", subj_id)
        obj_text  = entities_map.get(obj_id,  {}).get("text", obj_id)

        conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
        print(f"    {C.GREEN}{subj_text}{C.RESET} "
              f"{C.CYAN}—[{rel}]→{C.RESET} "
              f"{C.YELLOW}{obj_text}{C.RESET} "
              f"{C.GRAY}conf={conf_str}{C.RESET}")
        if evidence:
            print(f"      {C.GRAY}  ↳ «{evidence}»{C.RESET}")


def compare_entities(llm_ents: list, gold_ents: list):
    """Упрощённое сравнение по тексту сущности."""
    print(f"\n  {C.BOLD}📊 СРАВНЕНИЕ СУЩНОСТЕЙ:{C.RESET}")

    llm_texts  = {e["text"].lower(): e for e in llm_ents}
    gold_texts = {e["text"].lower(): e for e in gold_ents}

    matched   = set(llm_texts) & set(gold_texts)
    llm_only  = set(llm_texts) - set(gold_texts)
    gold_only = set(gold_texts) - set(llm_texts)

    # Совпадения с проверкой класса
    class_mismatches = []
    for t in matched:
        if llm_texts[t].get("class") != gold_texts[t].get("class"):
            class_mismatches.append((t, llm_texts[t]["class"], gold_texts[t]["class"]))

    print(f"    {C.GREEN}✓ Совпадений: {len(matched)}{C.RESET}")
    if class_mismatches:
        print(f"    {C.YELLOW}⚠ Расхождения в классе ({len(class_mismatches)}):{C.RESET}")
        for t, llm_cls, gold_cls in class_mismatches:
            print(f"      «{t}»: LLM={llm_cls}, Gold={gold_cls}")

    if llm_only:
        print(f"    {C.RED}+ Только у LLM ({len(llm_only)}): {C.RESET}"
              + ", ".join(f"«{t}»" for t in llm_only))

    if gold_only:
        print(f"    {C.BLUE}− Только в Gold ({len(gold_only)}): {C.RESET}"
              + ", ".join(f"«{t}»" for t in gold_only))

    # Простая F1
    if len(matched) + len(llm_only) > 0 and len(matched) + len(gold_only) > 0:
        precision = len(matched) / (len(matched) + len(llm_only))
        recall    = len(matched) / (len(matched) + len(gold_only))
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\n    {C.BOLD}Precision: {precision:.2f}  Recall: {recall:.2f}  F1: {f1:.2f}{C.RESET}")


def review_doc(doc_id: str, gold_docs: dict, compare: bool = False):
    ann = load_annotation(doc_id)
    if ann is None:
        print(f"{C.RED}Разметка для {doc_id} не найдена в {ANNOTATIONS_DIR}{C.RESET}")
        return

    print_header(f"Документ: {doc_id} | {ann.get('product', '?')}")
    print(f"\n  {C.BOLD}ТЕКСТ:{C.RESET}")
    # Вывод текста с переносом
    words = ann["text"].split()
    line  = "  "
    for w in words:
        if len(line) + len(w) > 70:
            print(line)
            line = "  " + w + " "
        else:
            line += w + " "
    if line.strip():
        print(line)

    if ann.get("status") == "error":
        print(f"\n  {C.RED}ОШИБКА РАЗМЕТКИ: {ann.get('error')}{C.RESET}")
        return

    annotation = ann.get("annotation", {})
    entities   = annotation.get("entities", [])
    relations  = annotation.get("relations", [])
    triples    = annotation.get("ontology_triples", [])
    notes      = annotation.get("annotation_notes", "")

    ent_map = {e["id"]: e for e in entities}

    # ── LLM разметка ────────────────────────────────────────────────
    print_entities(entities, "🤖 LLM-разметка")
    print_relations(relations, ent_map, "🤖 LLM-отношения")

    if triples:
        print(f"\n  {C.BOLD}🔗 OWL-триплеты:{C.RESET}")
        for t in triples:
            print(f"    {C.GRAY}{t}{C.RESET}")

    if notes:
        print(f"\n  {C.BOLD}📝 Заметки LLM:{C.RESET}")
        print(f"    {C.GRAY}{notes}{C.RESET}")

    # ── Gold standard ────────────────────────────────────────────────
    if doc_id in gold_docs:
        gold = gold_docs[doc_id]
        gold_ents = gold.get("entities", [])
        gold_rels = gold.get("relations", [])
        gold_ent_map = {e["id"]: e for e in gold_ents}

        print(f"\n  {'─'*50}")
        print_entities(gold_ents, "✅ Gold standard")
        print_relations(gold_rels, gold_ent_map, "✅ Gold отношения")

        if compare:
            compare_entities(entities, gold_ents)

    elif compare:
        print(f"\n  {C.GRAY}[Gold standard для {doc_id} не найден]{C.RESET}")

    # ── Форма для ручной проверки ────────────────────────────────────
    print(f"\n  {C.BOLD}📋 ТВОЯ ОЦЕНКА:{C.RESET}")
    print(f"  {C.GRAY}Поставь оценку каждой сущности (корректно / неверный класс / лишняя / пропущена){C.RESET}")
    print(f"  {C.GRAY}Результаты запиши в: annotations/{doc_id}_review.json{C.RESET}")

    # Шаблон файла для ревью
    review_template = {
        "doc_id":   doc_id,
        "reviewer": "human",
        "entities_review": [
            {
                "id":      e["id"],
                "text":    e["text"],
                "llm_class": e.get("class"),
                "verdict": "ok | wrong_class | spurious",
                "correct_class": None,
                "comment": ""
            }
            for e in entities
        ],
        "missing_entities": [],
        "relations_review": [
            {
                "id":      r["id"],
                "verdict": "ok | wrong_relation | spurious",
                "comment": ""
            }
            for r in relations
        ],
        "overall_score": None,
        "notes": ""
    }

    review_path = ANNOTATIONS_DIR / f"{doc_id}_review.json"
    if not review_path.exists():
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(review_template, f, ensure_ascii=False, indent=2)
        print(f"  {C.GREEN}→ Шаблон ревью создан: {review_path.name}{C.RESET}")
    else:
        print(f"  {C.YELLOW}→ Ревью уже существует: {review_path.name}{C.RESET}")


def main():
    parser = argparse.ArgumentParser(description="Ручная проверка LLM-разметки")
    parser.add_argument("--doc", nargs="*", help="ID документов (по умолчанию — все размеченные)")
    parser.add_argument("--compare", action="store_true", help="Сравнить с gold standard")
    args = parser.parse_args()

    gold_docs = load_gold()

    if args.doc:
        doc_ids = args.doc
    else:
        doc_ids = sorted([
            p.stem.replace("_annotation", "")
            for p in ANNOTATIONS_DIR.glob("*_annotation.json")
        ])

    if not doc_ids:
        print(f"{C.YELLOW}Размеченных документов не найдено в {ANNOTATIONS_DIR}{C.RESET}")
        print("Сначала запусти: python auto_annotate.py")
        return

    for doc_id in doc_ids:
        review_doc(doc_id, gold_docs, compare=args.compare)
        print()


if __name__ == "__main__":
    main()
