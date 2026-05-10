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
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass, field

from baseline.baseline_ner import SpacyFinancialNER, Entity
from baseline.ontology_graph import OntologyGraph, fin_uri
from evaluate import evaluate
from corpus.loader import load_corpus, load_gold

# pymorphy3 — для приведения текста сущностей к именительному падежу.
# Тот же подход что в агентной системе: лемматизация служит ключом
# дедупликации и нормализует написание ("заёмщика" → "заёмщик").
try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
except ImportError:
    _morph = None


@lru_cache(maxsize=20000)
def _normalize_text(text: str) -> str:
    """
    Лемматизирует текст пословно. Используется и как ключ дедупликации,
    и как канонический текст сущности. Если pymorphy3 недоступен —
    возвращает text.lower() (graceful fallback).
    """
    if not _morph:
        return text.lower().strip()
    parts = re.findall(r"\w+|[^\w\s]+|\s+", text)
    out = []
    for p in parts:
        if p.strip() and p[0].isalpha():
            out.append(_morph.parse(p)[0].normal_form)
        else:
            out.append(p)
    return "".join(out).strip()


OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ИЗВЛЕЧЕНИЕ ОТНОШЕНИЙ
# Простые правила на основе типов пар сущностей + текста между ними
# ─────────────────────────────────────────────────────────────────────────────

# Текстовые паттерны между двумя сущностями → тип отношения.
# Покрывают активные, страдательные и юридические конструкции.
# Порядок важен: более специфичные паттерны идут раньше — re.search
# берёт первое совпадение в списке.
BETWEEN_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Юридические конструкции — частотны в законах, типовых договорах.
    # Идут первыми: "подлежит обязательному..." должно матчиться как appliesTo,
    # а не как requires из-за слова "обязательному" дальше.
    (re.compile(r'признаётся|признается|считается|является',                              re.I), "subClassOf"),
    (re.compile(r'подлежит|применяется|распространяется',                                 re.I), "appliesTo"),

    # Активные конструкции
    (re.compile(r'предоставляет|выдаёт|выдает|открывает|эмитирует|размещает|оформляет',   re.I), "issuedBy"),
    (re.compile(r'регулируется|устанавливает|осуществляет надзор|контролирует',           re.I), "regulatedBy"),
    (re.compile(r'обязан|необходимо|обязательн',                                          re.I), "requires"),
    (re.compile(r'осуществляет|производит|допускается|включает',                          re.I), "involves"),
    (re.compile(r'составляет|равн[аео]|установлен',                                       re.I), "hasValue"),

    # Страдательные конструкции — типичны в документах банков и нормативке.
    # Используем `(е|ю)тся` чтобы матчить и ед.ч. и мн.ч.: "начисляется"/"начисляются"
    (re.compile(r'выпуска(е|ю)тся|оформля(е|ю)тся|предоставля(е|ю)тся|размеща(е|ю)тся|выда[её]тся', re.I), "issuedBy"),
    (re.compile(r'начисля(е|ю)тся|производ(и|я)тся|осуществля(е|ю)тся',                             re.I), "involves"),
    (re.compile(r'требу(е|ю)тся|необходим[оа]?|должен быть',                                        re.I), "requires"),
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
                    continue

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
        # Отношения из extract_relations уже имеют subject_text/object_text —
        # переводим в формат gold: subject/object это id сущностей
        ent_text_to_id = {e.text: e.id for e in self.entities}
        relations_gold = []
        for i, r in enumerate(self.relations, 1):
            relations_gold.append({
                "id":         r.get("id", f"r{i}"),
                "subject":    ent_text_to_id.get(r.get("subject_text", ""), r.get("subject", "")),
                "relation":   r["relation"],
                "object":     ent_text_to_id.get(r.get("object_text", ""), r.get("object", "")),
                "confidence": r.get("confidence", 0.0),
                "evidence":   r.get("evidence", ""),
            })
        return {
            "id":                self.doc_id,
            "title":             self.product,
            "source":            self.doc_id + ".txt",
            "entities":          [e.to_dict() for e in self.entities],
            "relations":         relations_gold,
            "ontology_triples":  [],
            "annotation_notes":  "",
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

        raw_entities = self.ner.extract(text)
        entities     = self._normalize_and_dedup(raw_entities)
        relations    = extract_relations(text, entities)
        relations    = self._dedup_relations(relations)

        if verbose:
            by_cls = {}
            for e in entities:
                by_cls[e.cls] = by_cls.get(e.cls, 0) + 1
            print(f"  [{doc_id}] {product}")
            print(f"    → {len(raw_entities)} сырых, {len(entities)} после дедупликации: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(by_cls.items())))
            print(f"    → {len(relations)} уникальных отношений")

        parent = DOC_TYPE_TO_PARENT.get(doc_type, "FinancialProduct")
        self._populate_graph(entities, relations, parent)

        return ProcessedDoc(doc_id, doc_type, product, text, entities, relations)

    @staticmethod
    def _normalize_and_dedup(entities: list[Entity]) -> list[Entity]:
        """
        Нормализует текст сущностей в именительный падеж и схлопывает дубли
        по ключу (нормализованный_текст, class, spacy_label).

        Это закрывает сразу две проблемы:
          - "заёмщика" / "заёмщиком" / "заёмщик" → одна сущность "заёмщик"
          - повторные совпадения ruler-паттернов на одно и то же понятие
        Перенумеровывает id чтобы они оставались последовательными.
        """
        seen: dict[tuple[str, str, str], Entity] = {}
        for e in entities:
            normalized = _normalize_text(e.text)
            if not normalized:
                continue
            key = (normalized, e.cls, e.spacy_label)
            if key in seen:
                continue
            # Сохраняем сущность с уже нормализованным text
            e.text = normalized
            seen[key] = e

        result = list(seen.values())
        for i, e in enumerate(result, 1):
            e.id = f"e{i}"
        return result

    @staticmethod
    def _dedup_relations(relations: list[dict]) -> list[dict]:
        """
        Убирает повторяющиеся тройки (subject_text, relation, object_text).
        Одна и та же связь часто выводится несколько раз — из разных
        предложений или из правил-перекрытий. В графе достаточно одной.
        """
        seen: set[tuple[str, str, str]] = set()
        unique: list[dict] = []
        for r in relations:
            key = (r.get("subject_text", ""), r["relation"], r.get("object_text", ""))
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        return unique

    def _populate_graph(self, entities, relations, parent):
        for e in entities:
            if e.cls == "FinancialProduct":
                self.graph.add_class(e.text, parent=parent)
            else:
                self.graph.add_instance(e.text, cls=e.cls)

        ent_map = {e.id: e for e in entities}
        for r in relations:
            e1 = ent_map.get(r["subject"])
            e2 = ent_map.get(r["object"])
            if e1 and e2:
                self.graph.add_relation(
                    subj=fin_uri(e1.text),
                    pred=r["relation"],
                    obj=fin_uri(e2.text),
                )

    def process_corpus(self, docs: list[dict], verbose: bool = True) -> list[ProcessedDoc]:
        return [self.process_doc(doc, verbose=verbose) for doc in docs]

    def save(self, results: list[ProcessedDoc]):
        self.graph.save_turtle(str(OUTPUT_DIR / "baseline_ontology.ttl"))
        self.graph.save_owl   (str(OUTPUT_DIR / "baseline_ontology.owl"))

        # Имя файла: один документ → его_id.json, несколько → results.json
        if len(results) == 1:
            ann_path = OUTPUT_DIR / f"{results[0].doc_id}.json"
        else:
            ann_path = OUTPUT_DIR / "results.json"

        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(
                {"documents": [r.to_dict() for r in results]},
                f, ensure_ascii=False, indent=2,
            )
        print(f"  Аннотации: {ann_path}")


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
    docs    = load_corpus(doc_ids=args.doc)

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

    if args.eval:
        gold      = load_gold()
        gold_path = OUTPUT_DIR / "gold_annotations.json"
        with open(gold_path, "w", encoding="utf-8") as f:
            json.dump(gold, f, ensure_ascii=False, indent=2)

        m = evaluate(
            results=results,
            gold_path=gold_path,
            get_entities=lambda r: [
                {"text": e.text, "class": e.cls} for e in r.entities
            ],
            get_relations=lambda r: r.relations,
            get_doc_id=lambda r: r.doc_id,
        )

        _print_metrics(m)

        eval_path = OUTPUT_DIR / "baseline_eval.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        print(f"  Метрики: {eval_path}")

    print(f"\n{'='*55}\n")


def _print_metrics(m: dict):
    print(f"\n{'─'*55}  ОЦЕНКА\n")

    for doc_id, dm in m["per_doc"].items():
        print(f"  [{doc_id}]")
        ep = dm["entity_partial"]
        es = dm["entity_strict"]
        er = dm["relation"]
        print(f"    entity partial  P={ep['precision']}  R={ep['recall']}  F1={ep['f1']}"
              f"  (TP={ep['tp']} FP={ep['fp']} FN={ep['fn']})")
        print(f"    entity strict   P={es['precision']}  R={es['recall']}  F1={es['f1']}"
              f"  (TP={es['tp']} FP={es['fp']} FN={es['fn']})")
        print(f"    relation        P={er['precision']}  R={er['recall']}  F1={er['f1']}"
              f"  (TP={er['tp']} FP={er['fp']} FN={er['fn']})")

        if es.get("class_errors"):
            print(f"    ошибки класса:")
            for ce in es["class_errors"]:
                print(f"      «{ce['text']}»: predicted={ce['predicted']}, gold={ce['gold']}")

        if es.get("per_class_f1"):
            print(f"    F1 по классам:")
            for cls, v in es["per_class_f1"].items():
                if v["tp"] + v["fp"] + v["fn"] > 0:
                    print(f"      {cls:<20} F1={v['f1']}  "
                          f"(TP={v['tp']} FP={v['fp']} FN={v['fn']})")

    ov = m["overall"]
    print(f"\n  ИТОГО (micro-avg)")
    for key, label in [("entity_partial", "entity partial"),
                        ("entity_strict",  "entity strict "),
                        ("relation",       "relation      ")]:
        v = ov[key]
        print(f"    {label}  P={v['micro_precision']}  "
              f"R={v['micro_recall']}  F1={v['micro_f1']}")
    print()


if __name__ == "__main__":
    main()
