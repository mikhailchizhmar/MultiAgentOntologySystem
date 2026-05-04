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
        self.logs.append(f"[{agent}] {message}")

    def to_dict(self) -> dict:
        return {
            "doc_id":   self.doc_id,
            "doc_type": self.doc_type,
            "product":  self.product,
            "terms":    [t.__dict__ for t in self.terms],
            "entities": [e.__dict__ for e in self.entities],
            "relations":[r.__dict__ for r in self.relations],
            "validated_triples": self.validated_triples,
            "errors":   self.errors,
            "logs":     self.logs,
        }
