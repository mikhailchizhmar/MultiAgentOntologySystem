"""
baseline_pipeline.py
─────────────────────
Baseline пайплайн: spaCy NER → rule-based отношения → rdflib OWL-граф.

Это Baseline 1 в сравнительной таблице диссертации.
Намеренно простой — никакого ML/LLM, только правила.

Запуск:
    python baseline_pipeline.py               # весь корпус
    python baseline_pipeline.py --doc doc_001 # один документ
    python baseline_pipeline.py --eval        # + оценка по gold standard

Зависимости:
    pip install spacy rdflib
    python -m spacy download ru_core_news_sm
"""

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field

from corpus.documents import CORPUS
from baseline.baseline_ner import SpacyFinancialNER, Entity
from baseline.ontology_graph import OntologyGraph, fin_uri

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ИЗВЛЕЧЕНИЕ ОТНОШЕНИЙ
# Простые правила на основе типов пар сущностей + текста между ними
# ─────────────────────────────────────────────────────────────────────────────

# Текстовые паттерны между двумя сущностями → тип отношения
BETWEEN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'предоставляет|выдаёт|выдает|открывает|эмитирует|размещает', re.I), "issuedBy"),
    (re.compile(r'обязан|необходимо|требуется|обязательн',                    re.I), "requires"),
    (re.compile(r'осуществляет|производит|допускается',                        re.I), "involves"),
    (re.compile(r'регулируется|устанавливает|осуществляет надзор',             re.I), "regulatedBy"),
    (re.compile(r'составляет|равн[аео]|установлен',                            re.I), "hasValue"),
]

# Матрица: (класс субъекта, класс объекта) → отношение по умолчанию
CLASS_PAIR_MAP: dict[tuple[str, str], str] = {
    ("FinancialProduct", "ProductAttribute"): "hasAttribute",
    ("FinancialProduct", "Process"):          "involves",
    ("FinancialProduct", "Condition"):        "requires",
    ("FinancialProduct", "Actor"):            "issuedBy",
    ("ProductAttribute", "Metric"):           "hasValue",
    ("Actor",            "FinancialProduct"): "issuedBy",
}


def extract_relations(text: str, entities: list[Entity]) -> list[dict]:
    """
    Извлекает отношения между сущностями в пределах одного предложения.
    Два прохода:
      1. Текстовые паттерны между сущностями
      2. Матрица типов пар (fallback)
    """
    relations = []
    counter   = 0

    # Разбиваем на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    pos = 0
    for sent in sentences:
        sent_start = text.find(sent, pos)
        sent_end   = sent_start + len(sent)
        pos        = sent_end

        sent_ents = [e for e in entities if sent_start <= e.start < sent_end]
        if len(sent_ents) < 2:
            continue

        for i, e1 in enumerate(sent_ents):
            for e2 in sent_ents[i + 1:]:
                if e1.end > e2.start:
                    continue  # перекрытие

                between = text[e1.end:e2.start]

                # Проход 1: текстовые паттерны
                rel_type = None
                for pattern, rtype in BETWEEN_PATTERNS:
                    if pattern.search(between):
                        rel_type = rtype
                        break

                # Проход 2: матрица типов
                if rel_type is None:
                    rel_type = CLASS_PAIR_MAP.get((e1.cls, e2.cls))

                if rel_type:
                    counter += 1
                    relations.append({
                        "id":           f"r{counter}",
                        "subject":      e1.id,
                        "subject_text": e1.text,
                        "relation":     rel_type,
                        "object":       e2.id,
                        "object_text":  e2.text,
                        "confidence":   0.65,
                        "evidence":     sent.strip(),
                    })

    return relations


# ─────────────────────────────────────────────────────────────────────────────
# МАППИНГ ТИПА ДОКУМЕНТА → РОДИТЕЛЬСКИЙ КЛАСС В ОНТОЛОГИИ
# ─────────────────────────────────────────────────────────────────────────────

DOC_TYPE_TO_PARENT = {
    "credit":     "CreditProduct",
    "deposit":    "DepositProduct",
    "investment": "InvestmentProduct",
    "insurance":  "InsuranceProduct",
    "card":       "CardProduct",
    "regulatory": "FinancialProduct",
}


# ─────────────────────────────────────────────────────────────────────────────
# ОСНОВНОЙ ПАЙПЛАЙН
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessedDoc:
    doc_id:    str
    doc_type:  str
    product:   str
    text:      str
    entities:  list[Entity]  = field(default_factory=list)
    relations: list[dict]    = field(default_factory=list)

    def to_dict(self) -> dict:
        by_class = {}
        for e in self.entities:
            by_class[e.cls] = by_class.get(e.cls, 0) + 1
        return {
            "id":        self.doc_id,
            "doc_type":  self.doc_type,
            "product":   self.product,
            "entities":  [e.to_dict() for e in self.entities],
            "relations": self.relations,
            "stats": {
                "entities":  len(self.entities),
                "relations": len(self.relations),
                "by_class":  by_class,
            },
        }


class BaselinePipeline:
    """
    spaCy NER → rule-based Relations → rdflib OWL-граф.
    """

    def __init__(self, spacy_model: str = "ru_core_news_sm"):
        self.ner   = SpacyFinancialNER(model=spacy_model)
        self.graph = OntologyGraph()

    def process_doc(self, doc: dict, verbose: bool = True) -> ProcessedDoc:
        doc_id   = doc["id"]
        text     = doc["text"]
        doc_type = doc["type"]
        product  = doc["product"]

        # ── NER ──────────────────────────────────────────────────────────────
        entities  = self.ner.extract(text)

        # ── Relations ────────────────────────────────────────────────────────
        relations = extract_relations(text, entities)

        if verbose:
            by_cls = {}
            for e in entities:
                by_cls[e.cls] = by_cls.get(e.cls, 0) + 1
            print(f"  [{doc_id}] {product}")
            print(f"    → {len(entities)} сущностей: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(by_cls.items())))
            print(f"    → {len(relations)} отношений")

        # ── Граф ─────────────────────────────────────────────────────────────
        parent = DOC_TYPE_TO_PARENT.get(doc_type, "FinancialProduct")
        self._populate_graph(entities, relations, parent)

        return ProcessedDoc(doc_id, doc_type, product, text, entities, relations)

    def _populate_graph(
        self,
        entities:  list[Entity],
        relations: list[dict],
        parent:    str,
    ):
        # Сущности → классы / экземпляры в графе
        for e in entities:
            if e.cls == "FinancialProduct":
                uri = self.graph.add_class(e.text, parent=parent)
            else:
                uri = self.graph.add_instance(e.text, cls=e.cls)

        # Отношения → триплеты
        ent_map = {}  # id → Entity
        for e in entities:
            ent_map[e.id] = e

        for r in relations:
            e1 = ent_map.get(r["subject"])
            e2 = ent_map.get(r["object"])
            if not e1 or not e2:
                continue
            self.graph.add_relation(
                subj=fin_uri(e1.text),
                pred=r["relation"],
                obj=fin_uri(e2.text),
            )

    def process_corpus(self, docs: list[dict], verbose: bool = True) -> list[ProcessedDoc]:
        results = []
        for doc in docs:
            results.append(self.process_doc(doc, verbose=verbose))
        return results

    def save(self, results: list[ProcessedDoc]):
        # Онтология
        self.graph.save_turtle(str(OUTPUT_DIR / "baseline_ontology.ttl"))
        self.graph.save_owl   (str(OUTPUT_DIR / "baseline_ontology.owl"))

        # Аннотации
        ann_path = OUTPUT_DIR / "baseline_annotations.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta":      {"system": "spacy_rule_based"},
                    "documents": [r.to_dict() for r in results],
                    "ontology_stats": self.graph.stats(),
                },
                f, ensure_ascii=False, indent=2,
            )
        print(f"  Аннотации: {ann_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ОЦЕНКА ПО GOLD STANDARD
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(results: list[ProcessedDoc], gold_path: Path) -> dict:
    """Precision / Recall / F1 по сущностям (exact match по тексту)."""
    with open(gold_path, encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_by_id = {d["id"]: d for d in gold_data["documents"]}
    metrics    = {}
    total_tp = total_fp = total_fn = 0

    for result in results:
        if result.doc_id not in gold_by_id:
            continue

        gold_ents = {e["text"].lower(): e["class"]
                     for e in gold_by_id[result.doc_id]["entities"]}
        pred_ents = {e.text.lower(): e.cls
                     for e in result.entities}

        gold_set, pred_set = set(gold_ents), set(pred_ents)
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        total_tp += tp; total_fp += fp; total_fn += fn

        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0

        metrics[result.doc_id] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 3),
            "recall":    round(r, 3),
            "f1":        round(f1, 3),
            "missed":    sorted(gold_set - pred_set),
            "spurious":  sorted(pred_set - gold_set),
            "class_errors": [
                {"text": t, "gold": gold_ents[t], "pred": pred_ents[t]}
                for t in (gold_set & pred_set)
                if gold_ents[t] != pred_ents[t]
            ],
        }

    p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    return {
        "per_doc": metrics,
        "overall": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "micro_precision": round(p, 3),
            "micro_recall":    round(r, 3),
            "micro_f1":        round(f1, 3),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc",   nargs="*", help="ID документов (по умолчанию — все)")
    parser.add_argument("--eval",  action="store_true", help="Оценить по gold standard")
    parser.add_argument("--model", default="ru_core_news_sm", help="spaCy модель")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    docs    = [d for d in CORPUS if d["id"] in args.doc] if args.doc else CORPUS

    print(f"\n{'='*55}")
    print(f"  BASELINE: spaCy NER → rdflib OWL")
    print(f"  Модель: {args.model}  |  Документов: {len(docs)}")
    print(f"{'='*55}\n")

    pipeline = BaselinePipeline(spacy_model=args.model)
    results  = pipeline.process_corpus(docs, verbose=verbose)

    stats = pipeline.graph.stats()
    print(f"\n  Онтология: {stats['classes']} классов, "
          f"{stats['properties']} свойств, {stats['triples']} триплетов")

    pipeline.save(results)

    # ── Оценка ───────────────────────────────────────────────────────────────
    if args.eval:
        gold_path = Path(__file__).parent.parent / "corpus" / "gold_annotations.json"
        if not gold_path.exists():
            print(f"\n  Gold standard не найден: {gold_path}")
            return

        m = evaluate(results, gold_path)

        print(f"\n{'─'*55}  ОЦЕНКА\n")
        for doc_id, dm in m["per_doc"].items():
            print(f"  [{doc_id}]  "
                  f"P={dm['precision']}  R={dm['recall']}  F1={dm['f1']}  "
                  f"(TP={dm['tp']} FP={dm['fp']} FN={dm['fn']})")
            if dm["missed"]:
                print(f"    пропущено:  {dm['missed']}")
            if dm["spurious"]:
                print(f"    лишние:     {dm['spurious']}")
            if dm["class_errors"]:
                print(f"    ошибки класса: {dm['class_errors']}")

        ov = m["overall"]
        print(f"\n  micro-avg  "
              f"P={ov['micro_precision']}  "
              f"R={ov['micro_recall']}  "
              f"F1={ov['micro_f1']}\n")

        eval_path = OUTPUT_DIR / "baseline_eval.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        print(f"  Метрики: {eval_path}")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
