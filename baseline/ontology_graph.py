"""
ontology_graph.py
──────────────────
Построение OWL-онтологии через rdflib.

Установка:
    pip install rdflib

Сериализует в:
  - Turtle (.ttl)  — читается Protégé
  - RDF/XML (.owl) — стандартный OWL формат
  - JSON-LD        — для веб-интеграций
"""

from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
import re


# ─────────────────────────────────────────────────────────────────────────────
# ПРОСТРАНСТВА ИМЁН
# ─────────────────────────────────────────────────────────────────────────────

FIN  = Namespace("http://fin-ontology.org/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


# ─────────────────────────────────────────────────────────────────────────────
# УТИЛИТЫ
# ─────────────────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Русский текст → CamelCase для URI."""
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
# ГРАФ
# ─────────────────────────────────────────────────────────────────────────────

class OntologyGraph:
    """
    Обёртка над rdflib.Graph для построения OWL-онтологии
    финансовых продуктов.
    """

    def __init__(self):
        self.g = Graph()
        self.g.bind("fin",  FIN)
        self.g.bind("owl",  OWL)
        self.g.bind("rdfs", RDFS)
        self.g.bind("skos", SKOS)
        self.g.bind("xsd",  XSD)

        # Объявляем онтологию
        onto = FIN["FinancialProductsOntology"]
        self.g.add((onto, RDF.type,       OWL.Ontology))
        self.g.add((onto, RDFS.label,     Literal("Онтология финансовых продуктов", lang="ru")))
        self.g.add((onto, RDFS.comment,   Literal("Baseline-версия. Построена автоматически.", lang="ru")))

        # Строим базовую схему
        self._init_schema()

    # ── Базовая схема ─────────────────────────────────────────────────────────

    def _init_schema(self):
        """Объявляет верхнеуровневые классы и свойства."""

        # Топ-классы
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
            self.g.add((uri, RDF.type,   OWL.Class))
            self.g.add((uri, RDFS.label, Literal(label, lang="ru")))

        # Подклассы FinancialProduct
        product_subclasses = {
            "CreditProduct":     "Кредитный продукт",
            "DepositProduct":    "Депозитный продукт",
            "InvestmentProduct": "Инвестиционный продукт",
            "InsuranceProduct":  "Страховой продукт",
            "CardProduct":       "Карточный продукт",
        }
        for name, label in product_subclasses.items():
            uri = FIN[name]
            self.g.add((uri, RDF.type,          OWL.Class))
            self.g.add((uri, RDFS.label,         Literal(label, lang="ru")))
            self.g.add((uri, RDFS.subClassOf,    FIN["FinancialProduct"]))

        # Свойства (ObjectProperty)
        properties = [
            ("hasAttribute",  FIN["FinancialProduct"], FIN["ProductAttribute"], "имеет атрибут"),
            ("issuedBy",      FIN["FinancialProduct"], FIN["Actor"],            "выпускается"),
            ("requires",      FIN["FinancialProduct"], FIN["Condition"],        "требует"),
            ("involves",      FIN["FinancialProduct"], FIN["Process"],          "включает процесс"),
            ("hasValue",      FIN["ProductAttribute"], FIN["Metric"],           "имеет значение"),
            ("appliesTo",     None,                    FIN["FinancialProduct"], "применяется к"),
            ("regulatedBy",   FIN["FinancialProduct"], FIN["Actor"],            "регулируется"),
        ]
        for name, domain, range_, label in properties:
            uri = FIN[name]
            self.g.add((uri, RDF.type,   OWL.ObjectProperty))
            self.g.add((uri, RDFS.label, Literal(label, lang="ru")))
            if domain:
                self.g.add((uri, RDFS.domain, domain))
            if range_:
                self.g.add((uri, RDFS.range,  range_))

    # ── Добавление элементов ─────────────────────────────────────────────────

    def add_class(self, label: str, parent: str | None = None) -> URIRef:
        """Добавляет класс. parent — строка вида 'CreditProduct'."""
        uri = fin_uri(label)
        self.g.add((uri, RDF.type,   OWL.Class))
        self.g.add((uri, RDFS.label, Literal(label, lang="ru")))
        if parent:
            self.g.add((uri, RDFS.subClassOf, FIN[parent]))
        return uri

    def add_instance(
        self,
        label: str,
        cls: str,
        comment: str | None = None
    ) -> URIRef:
        """Добавляет экземпляр класса."""
        uri = fin_uri(label)
        self.g.add((uri, RDF.type,   FIN[cls]))
        self.g.add((uri, RDFS.label, Literal(label, lang="ru")))
        if comment:
            self.g.add((uri, RDFS.comment, Literal(comment, lang="ru")))
        return uri

    def add_relation(self, subj: URIRef, pred: str, obj: URIRef):
        """Добавляет тройку с предикатом из нашего пространства имён."""
        self.g.add((subj, FIN[pred], obj))

    # ── Статистика ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        n_classes = sum(
            1 for _ in self.g.subjects(RDF.type, OWL.Class)
        )
        n_props = sum(
            1 for _ in self.g.subjects(RDF.type, OWL.ObjectProperty)
        )
        return {
            "triples":    len(self.g),
            "classes":    n_classes,
            "properties": n_props,
        }

    # ── Сериализация ─────────────────────────────────────────────────────────

    def save_turtle(self, path: str):
        self.g.serialize(destination=path, format="turtle")
        print(f"  Сохранено (Turtle): {path}")

    def save_owl(self, path: str):
        self.g.serialize(destination=path, format="xml")
        print(f"  Сохранено (OWL/XML): {path}")

    def save_jsonld(self, path: str):
        self.g.serialize(destination=path, format="json-ld", indent=2)
        print(f"  Сохранено (JSON-LD): {path}")
