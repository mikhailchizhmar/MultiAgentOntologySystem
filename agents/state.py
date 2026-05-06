"""
state.py
─────────
Общее состояние мультиагентного пайплайна.

LangGraph передаёт этот объект между узлами-агентами.
Каждый агент читает нужные поля и дописывает свои результаты.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Term:
    """Термин-кандидат после TermExtractor."""
    text:       str
    context:    str          # предложение, в котором встретился термин
    score:      float        # статистический score (TF-IDF-подобный)
    source:     str          # "statistical" | "llm" | "both"
    frequency:  int = 1


@dataclass
class Entity:
    """Размеченная сущность после EntityClassifier."""
    id:         str
    text:       str
    cls:        str          # FinancialProduct | ProductAttribute | Actor | ...
    context:    str
    confidence: float
    comment:    str = ""


@dataclass
class Relation:
    """Отношение после RelationExtractor."""
    id:           str
    subject_id:   str
    subject_text: str
    relation:     str        # hasAttribute | issuedBy | requires | ...
    object_id:    str
    object_text:  str
    confidence:   float
    evidence:     str        # цитата из текста


@dataclass
class PipelineState:
    """
    Полное состояние пайплайна для одного документа.
    Передаётся между узлами LangGraph.
    """

    # ── Входные данные ────────────────────────────────────────────────
    doc_id:   str = ""
    doc_type: str = ""
    product:  str = ""
    text:     str = ""

    # ── Результаты агентов ────────────────────────────────────────────
    terms:             list[Term]     = field(default_factory=list)
    entities:          list[Entity]   = field(default_factory=list)
    relations:         list[Relation] = field(default_factory=list)
    validated_triples: list[dict]     = field(default_factory=list)

    # ── Метаданные и логи ─────────────────────────────────────────────
    errors:   list[str] = field(default_factory=list)
    logs:     list[str] = field(default_factory=list)

    def log(self, agent: str, message: str):
        entry = f"[{agent}] {message}"
        self.logs.append(entry)
        # Выводим сразу — так видно прогресс в реальном времени,
        # а не только после завершения всего документа
        print(f"  {entry}")

    def to_dict(self) -> dict:
        # Сущности — формат gold
        entities_gold = [
            {
                "id":         e.id,
                "text":       e.text,
                "class":      e.cls,
                "start":      None,
                "confidence": e.confidence,
                "comment":    e.comment,
            }
            for e in self.entities
        ]
        # Отношения — формат gold (subject/object — id сущностей)
        relations_gold = [
            {
                "id":         r.id,
                "subject":    r.subject_id,
                "relation":   r.relation,
                "object":     r.object_id,
                "confidence": r.confidence,
                "evidence":   r.evidence,
            }
            for r in self.relations
        ]
        # ontology_triples берём из validated_triples как текстовые строки
        triples = [
            f"{t['subject']} :{t['relation']} :{t['object']}"
            for t in self.validated_triples
        ]
        return {
            "id":               self.doc_id,
            "title":            self.product,
            "source":           self.doc_id + ".txt",
            "entities":         entities_gold,
            "relations":        relations_gold,
            "ontology_triples": triples,
            "annotation_notes": "",
        }
