"""
ontology_integrator.py
───────────────────────
Агент 5: OntologyIntegrator

Вход:  state.validated_triples + state.entities
Выход: обновлённый rdflib граф (передаётся снаружи, агент его дополняет)

Задачи:
  1. Зарегистрировать каждую сущность как класс или экземпляр в OWL
  2. Добавить валидные тройки как ObjectProperty assertions
  3. Попытаться смаппить на FIBO через косинусное сходство эмбеддингов
     (упрощённая версия: маппинг по ключевым словам без внешних моделей)
"""

from __future__ import annotations

import re
import sys
import json
import os
from pathlib import Path

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL

from agents.state import PipelineState, Entity

# ─────────────────────────────────────────────────────────────────────────────
# ПРОСТРАНСТВА ИМЁН
# ─────────────────────────────────────────────────────────────────────────────

FIN  = Namespace("http://fin-ontology.org/")
FIBO = Namespace("https://spec.edmcouncil.org/fibo/ontology/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# ─────────────────────────────────────────────────────────────────────────────
# МАППИНГ НАШЕГО КЛАССА → РОДИТЕЛЬСКИЙ КЛАСС В OWL-СХЕМЕ
# ─────────────────────────────────────────────────────────────────────────────

ONTO_CLASS_MAP = {
    "FinancialProduct":  FIN["FinancialProduct"],
    "ProductAttribute":  FIN["ProductAttribute"],
    "Actor":             FIN["Actor"],
    "Process":           FIN["Process"],
    "Condition":         FIN["Condition"],
    "LegalTerm":         FIN["LegalTerm"],
    "Metric":            FIN["Metric"],
}

DOC_TYPE_TO_SUBCLASS = {
    "credit":     FIN["CreditProduct"],
    "deposit":    FIN["DepositProduct"],
    "investment": FIN["InvestmentProduct"],
    "insurance":  FIN["InsuranceProduct"],
    "card":       FIN["CardProduct"],
    "regulatory": FIN["FinancialProduct"],
}

# ─────────────────────────────────────────────────────────────────────────────
# УПРОЩЁННЫЙ МАППИНГ НА FIBO (ключевые слова → FIBO URI)
# Полноценный маппинг через эмбеддинги — в следующей итерации
# ─────────────────────────────────────────────────────────────────────────────

FIBO_MAPPING: dict[str, str] = {
    "кредит":               "LOAN/Loan",
    "ипотечный кредит":     "LOAN/ResidentialMortgageLoan",
    "потребительский кредит":"LOAN/ConsumerLoan",
    "вклад":                "DEP/Deposit",
    "депозит":              "DEP/Deposit",
    "облигация":            "SEC/Bond",
    "пиф":                  "FUND/CollectiveInvestmentVehicle",
    "паевой инвестиционный фонд": "FUND/CollectiveInvestmentVehicle",
    "процентная ставка":    "LOAN/NominalInterestRate",
    "банк":                 "FBC/Bank",
    "страховщик":           "INS/Insurer",
    "управляющая компания": "FUND/FundManager",
}


# ─────────────────────────────────────────────────────────────────────────────
# УТИЛИТЫ
# ─────────────────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    _TR = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo','ж':'zh',
        'з':'z','и':'i','й':'y','к':'k','л':'l','м':'m','н':'n','о':'o',
        'п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f','х':'kh','ц':'ts',
        'ч':'ch','ш':'sh','щ':'sch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu',
        'я':'ya',
    }
    result = []
    for ch in text.lower():
        result.append(_TR.get(ch, ch))
    slug = ''.join(result)
    slug = re.sub(r'[^a-z0-9\s]', '', slug)
    return ''.join(w.capitalize() for w in slug.split()) or "Unknown"


def fin_uri(label: str) -> URIRef:
    return FIN[slugify(label)]


# ─────────────────────────────────────────────────────────────────────────────
# АГЕНТ
# ─────────────────────────────────────────────────────────────────────────────

class OntologyIntegratorAgent:
    """
    Записывает сущности и тройки из PipelineState в rdflib граф.

    graph передаётся снаружи — один граф накапливает результаты
    по всем документам корпуса.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def run(self, state: PipelineState) -> PipelineState:
        state.log("OntologyIntegrator",
                  f"Интегрирую: {len(state.entities)} сущностей, "
                  f"{len(state.validated_triples)} троек")

        try:
            parent_class = DOC_TYPE_TO_SUBCLASS.get(
                state.doc_type, FIN["FinancialProduct"]
            )

            # 1. Регистрируем сущности
            added_entities = 0
            for e in state.entities:
                self._add_entity(e, parent_class)
                added_entities += 1

            # 2. Добавляем валидные тройки
            added_triples = 0
            for triple in state.validated_triples:
                self._add_triple(triple)
                added_triples += 1

            # 3. FIBO-маппинг
            fibo_mappings = self._apply_fibo_mapping(state.entities)

            state.log("OntologyIntegrator",
                      f"Добавлено: {added_entities} сущностей, "
                      f"{added_triples} троек, "
                      f"{fibo_mappings} FIBO-маппингов")

        except Exception as e:
            state.errors.append(f"OntologyIntegrator: {e}")
            state.log("OntologyIntegrator", f"ОШИБКА: {e}")

        return state

    # ── Вспомогательные методы ───────────────────────────────────────────────

    def _add_entity(self, e: Entity, parent_class: URIRef):
        """Добавляет сущность в граф как OWL-класс или экземпляр."""
        uri        = fin_uri(e.text)
        top_class  = ONTO_CLASS_MAP.get(e.cls, FIN["FinancialProduct"])

        if e.cls == "FinancialProduct":
            # FinancialProduct → OWL Class
            self.graph.add((uri, RDF.type,        OWL.Class))
            self.graph.add((uri, RDFS.subClassOf,  parent_class))
        else:
            # Всё остальное → именованный экземпляр
            self.graph.add((uri, RDF.type, top_class))

        # Метка на русском
        self.graph.add((uri, RDFS.label, Literal(e.text, lang="ru")))

        # Уверенность как аннотация (для отчётности)
        self.graph.add((
            uri,
            FIN["confidence"],
            Literal(str(round(e.confidence, 3)))
        ))

    def _add_triple(self, triple: dict):
        """Добавляет отношение в граф."""
        s   = fin_uri(triple["subject"])
        p   = FIN[triple["relation"]]
        o   = fin_uri(triple["object"])

        self.graph.add((s, p, o))

        # Провенанс (evidence) — как аннотация на реификацию
        # Упрощённо: храним в отдельном узле
        if triple.get("evidence"):
            evidence_node = FIN[f"ev_{slugify(triple['subject'])}_{triple['relation']}"]
            self.graph.add((evidence_node, RDF.type,    FIN["Evidence"]))
            self.graph.add((evidence_node, FIN["about"], s))
            self.graph.add((evidence_node, RDFS.comment,
                            Literal(triple["evidence"][:200], lang="ru")))

    def _apply_fibo_mapping(self, entities: list[Entity]) -> int:
        """
        Добавляет skos:exactMatch или skos:closeMatch на FIBO-классы.
        Возвращает количество созданных маппингов.
        """
        count = 0
        for e in entities:
            fibo_path = FIBO_MAPPING.get(e.text.lower())
            if fibo_path:
                uri      = fin_uri(e.text)
                fibo_uri = FIBO[fibo_path]
                self.graph.add((uri, SKOS.closeMatch, fibo_uri))
                count += 1
        return count

    # ── Сохранение графа ─────────────────────────────────────────────────────

    def save(self, output_dir: str):
        """Сохраняет граф в Turtle и OWL/XML."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ttl = out / "ontology.ttl"
        owl = out / "ontology.owl"

        self.graph.serialize(destination=str(ttl), format="turtle")
        self.graph.serialize(destination=str(owl), format="xml")

        print(f"  Граф сохранён: {ttl} ({len(self.graph)} триплетов)")
        print(f"  Граф сохранён: {owl}")

    def stats(self) -> dict:
        return {
            "total_triples": len(self.graph),
            "classes":  sum(1 for _ in self.graph.subjects(RDF.type, OWL.Class)),
            "instances": sum(
                1 for _ in self.graph.subjects(RDF.type, None)
                if (None, RDF.type, OWL.Class) not in self.graph
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ИНИЦИАЛИЗАЦИЯ БАЗОВОЙ СХЕМЫ ГРАФА
# (вызывается один раз перед запуском пайплайна)
# ─────────────────────────────────────────────────────────────────────────────

def build_base_graph() -> Graph:
    """Создаёт rdflib граф с базовой OWL-схемой."""
    g = Graph()
    g.bind("fin",  FIN)
    g.bind("fibo", FIBO)
    g.bind("skos", SKOS)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)

    # Онтология
    onto = FIN["FinancialProductsOntology"]
    g.add((onto, RDF.type,     OWL.Ontology))
    g.add((onto, RDFS.label,   Literal("Онтология финансовых продуктов", lang="ru")))
    g.add((onto, RDFS.comment, Literal("Мультиагентная система. GPT-4o-mini + LangGraph.", lang="ru")))

    # Верхнеуровневые классы
    top_classes = {
        "FinancialProduct":  "Финансовый продукт",
        "ProductAttribute":  "Атрибут продукта",
        "Actor":             "Участник",
        "Process":           "Процесс",
        "Condition":         "Условие",
        "LegalTerm":         "Юридический термин",
        "Metric":            "Метрика",
    }
    for name, label in top_classes.items():
        uri = FIN[name]
        g.add((uri, RDF.type,   OWL.Class))
        g.add((uri, RDFS.label, Literal(label, lang="ru")))

    # Подклассы продуктов
    for name, label in {
        "CreditProduct":     "Кредитный продукт",
        "DepositProduct":    "Депозитный продукт",
        "InvestmentProduct": "Инвестиционный продукт",
        "InsuranceProduct":  "Страховой продукт",
        "CardProduct":       "Карточный продукт",
    }.items():
        uri = FIN[name]
        g.add((uri, RDF.type,        OWL.Class))
        g.add((uri, RDFS.label,      Literal(label, lang="ru")))
        g.add((uri, RDFS.subClassOf, FIN["FinancialProduct"]))

    # ObjectProperty
    for name, domain, range_, label in [
        ("hasAttribute", FIN["FinancialProduct"],  FIN["ProductAttribute"], "имеет атрибут"),
        ("issuedBy",     FIN["FinancialProduct"],  FIN["Actor"],            "выпускается"),
        ("requires",     FIN["FinancialProduct"],  FIN["Condition"],        "требует"),
        ("involves",     FIN["FinancialProduct"],  FIN["Process"],          "включает процесс"),
        ("hasValue",     FIN["ProductAttribute"],  FIN["Metric"],           "имеет значение"),
        ("appliesTo",    None,                     FIN["FinancialProduct"], "применяется к"),
        ("regulatedBy",  FIN["FinancialProduct"],  FIN["Actor"],            "регулируется"),
        ("subClassOf",   FIN["FinancialProduct"],  FIN["FinancialProduct"], "подтип"),
    ]:
        uri = FIN[name]
        g.add((uri, RDF.type,   OWL.ObjectProperty))
        g.add((uri, RDFS.label, Literal(label, lang="ru")))
        if domain:
            g.add((uri, RDFS.domain, domain))
        if range_:
            g.add((uri, RDFS.range,  range_))

    return g
