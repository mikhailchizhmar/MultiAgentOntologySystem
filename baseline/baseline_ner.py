"""
baseline_ner.py
───────────────
Baseline NER на spaCy для финансовых текстов на русском языке.

Установка:
    pip install spacy
    python -m spacy download ru_core_news_sm

Стратегия:
    1. spaCy встроенный NER — ловит PER, ORG, LOC, MONEY, DATE и т.д.
    2. EntityRuler поверх — добавляет финансово-доменные паттерны,
       которых в базовой модели нет (ProcessentRate, FinancialProduct и т.д.)
    3. Маппинг spaCy-меток → наши классы онтологии
"""

import re
import spacy
from spacy.pipeline import EntityRuler
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# МАППИНГ: spaCy метки → классы онтологии
# ─────────────────────────────────────────────────────────────────────────────

SPACY_TO_ONTOLOGY = {
    # Стандартные метки spaCy (ru_core_news_sm)
    "ORG":      "Actor",           # организации: банк, УК, биржа
    "PER":      "Actor",           # персоны (редко в финдоках)
    "LOC":      "Actor",           # иногда ЦБ РФ распознаётся как LOC
    "MONEY":    "Metric",          # денежные суммы
    "PERCENT":  "Metric",          # проценты
    "DATE":     "Metric",          # сроки и даты
    "CARDINAL": "Metric",          # числа без единиц

    # Наши кастомные метки (из EntityRuler ниже)
    "FIN_PRODUCT":   "FinancialProduct",
    "FIN_ATTRIBUTE": "ProductAttribute",
    "FIN_PROCESS":   "Process",
    "FIN_CONDITION": "Condition",
    "FIN_LEGAL":     "LegalTerm",
    "FIN_METRIC":    "Metric",
    "FIN_ACTOR":     "Actor",
}

# Классы онтологии (для справки)
ENTITY_CLASSES = [
    "FinancialProduct",
    "ProductAttribute",
    "Actor",
    "Process",
    "Condition",
    "LegalTerm",
    "Metric",
]


# ─────────────────────────────────────────────────────────────────────────────
# ПАТТЕРНЫ ДЛЯ EntityRuler
# Формат: {"label": "...", "pattern": "строка"} или с токен-паттернами
# ─────────────────────────────────────────────────────────────────────────────

RULER_PATTERNS = [

    # ── FinancialProduct ─────────────────────────────────────────────────────
    {"label": "FIN_PRODUCT", "pattern": "ипотечный кредит"},
    {"label": "FIN_PRODUCT", "pattern": "ипотека"},
    {"label": "FIN_PRODUCT", "pattern": "потребительский кредит"},
    {"label": "FIN_PRODUCT", "pattern": "кредит наличными"},
    {"label": "FIN_PRODUCT", "pattern": "автокредит"},
    {"label": "FIN_PRODUCT", "pattern": "кредитная линия"},
    {"label": "FIN_PRODUCT", "pattern": "возобновляемая кредитная линия"},
    {"label": "FIN_PRODUCT", "pattern": "кредитная карта"},
    {"label": "FIN_PRODUCT", "pattern": "дебетовая карта"},
    {"label": "FIN_PRODUCT", "pattern": "срочный вклад"},
    {"label": "FIN_PRODUCT", "pattern": "накопительный счёт"},
    {"label": "FIN_PRODUCT", "pattern": "накопительный счет"},
    {"label": "FIN_PRODUCT", "pattern": "валютный вклад"},
    {"label": "FIN_PRODUCT", "pattern": "паевой инвестиционный фонд"},
    {"label": "FIN_PRODUCT", "pattern": "ПИФ"},
    {"label": "FIN_PRODUCT", "pattern": "инвестиционный пай"},
    {"label": "FIN_PRODUCT", "pattern": "облигация"},
    {"label": "FIN_PRODUCT", "pattern": "корпоративная облигация"},
    {"label": "FIN_PRODUCT", "pattern": "индивидуальный инвестиционный счёт"},
    {"label": "FIN_PRODUCT", "pattern": "ИИС"},
    {"label": "FIN_PRODUCT", "pattern": "полис КАСКО"},
    {"label": "FIN_PRODUCT", "pattern": "КАСКО"},
    {"label": "FIN_PRODUCT", "pattern": "открытый фонд облигаций"},
    {"label": "FIN_PRODUCT", "pattern": "транш"},
    {"label": "FIN_PRODUCT", "pattern": "вклад"},
    {"label": "FIN_PRODUCT", "pattern": "депозит"},

    # ── ProductAttribute ─────────────────────────────────────────────────────
    {"label": "FIN_ATTRIBUTE", "pattern": "процентная ставка"},
    {"label": "FIN_ATTRIBUTE", "pattern": "полная стоимость кредита"},
    {"label": "FIN_ATTRIBUTE", "pattern": "ПСК"},
    {"label": "FIN_ATTRIBUTE", "pattern": "аннуитетный платёж"},
    {"label": "FIN_ATTRIBUTE", "pattern": "аннуитетный платеж"},
    {"label": "FIN_ATTRIBUTE", "pattern": "ежемесячный платёж"},
    {"label": "FIN_ATTRIBUTE", "pattern": "минимальный ежемесячный платёж"},
    {"label": "FIN_ATTRIBUTE", "pattern": "срок кредитования"},
    {"label": "FIN_ATTRIBUTE", "pattern": "срок кредита"},
    {"label": "FIN_ATTRIBUTE", "pattern": "срок вклада"},
    {"label": "FIN_ATTRIBUTE", "pattern": "срок размещения"},
    {"label": "FIN_ATTRIBUTE", "pattern": "сумма кредита"},
    {"label": "FIN_ATTRIBUTE", "pattern": "первоначальный взнос"},
    {"label": "FIN_ATTRIBUTE", "pattern": "залог приобретаемой недвижимости"},
    {"label": "FIN_ATTRIBUTE", "pattern": "предмет залога"},
    {"label": "FIN_ATTRIBUTE", "pattern": "лимит кредитной линии"},
    {"label": "FIN_ATTRIBUTE", "pattern": "кредитный лимит"},
    {"label": "FIN_ATTRIBUTE", "pattern": "надбавка"},
    {"label": "FIN_ATTRIBUTE", "pattern": "скидка при погашении паёв"},
    {"label": "FIN_ATTRIBUTE", "pattern": "скидка при погашении"},
    {"label": "FIN_ATTRIBUTE", "pattern": "вознаграждение управляющей компании"},
    {"label": "FIN_ATTRIBUTE", "pattern": "минимальная сумма инвестиции"},
    {"label": "FIN_ATTRIBUTE", "pattern": "минимальная сумма"},
    {"label": "FIN_ATTRIBUTE", "pattern": "грейс-период"},
    {"label": "FIN_ATTRIBUTE", "pattern": "льготный период"},
    {"label": "FIN_ATTRIBUTE", "pattern": "франшиза"},
    {"label": "FIN_ATTRIBUTE", "pattern": "купонная ставка"},
    {"label": "FIN_ATTRIBUTE", "pattern": "купонный доход"},
    {"label": "FIN_ATTRIBUTE", "pattern": "страховая сумма"},
    {"label": "FIN_ATTRIBUTE", "pattern": "страховая премия"},
    {"label": "FIN_ATTRIBUTE", "pattern": "стоимость чистых активов"},
    {"label": "FIN_ATTRIBUTE", "pattern": "СЧА"},
    {"label": "FIN_ATTRIBUTE", "pattern": "кэшбэк"},
    {"label": "FIN_ATTRIBUTE", "pattern": "коэффициент бонус-малус"},

    # ── Actor ─────────────────────────────────────────────────────────────────
    {"label": "FIN_ACTOR", "pattern": "заёмщик"},
    {"label": "FIN_ACTOR", "pattern": "заемщик"},
    {"label": "FIN_ACTOR", "pattern": "кредитор"},
    {"label": "FIN_ACTOR", "pattern": "управляющая компания"},
    {"label": "FIN_ACTOR", "pattern": "пайщик"},
    {"label": "FIN_ACTOR", "pattern": "страховщик"},
    {"label": "FIN_ACTOR", "pattern": "страхователь"},
    {"label": "FIN_ACTOR", "pattern": "выгодоприобретатель"},
    {"label": "FIN_ACTOR", "pattern": "застрахованный"},
    {"label": "FIN_ACTOR", "pattern": "эмитент"},
    {"label": "FIN_ACTOR", "pattern": "организатор размещения"},
    {"label": "FIN_ACTOR", "pattern": "инвестиционный банк"},
    {"label": "FIN_ACTOR", "pattern": "брокер"},
    {"label": "FIN_ACTOR", "pattern": "Банк России"},
    {"label": "FIN_ACTOR", "pattern": "Центральный банк Российской Федерации"},
    {"label": "FIN_ACTOR", "pattern": "Московская биржа"},
    {"label": "FIN_ACTOR", "pattern": "страховая компания"},
    {"label": "FIN_ACTOR", "pattern": "аккредитованная страховая компания"},

    # ── Process ───────────────────────────────────────────────────────────────
    {"label": "FIN_PROCESS", "pattern": "досрочное погашение"},
    {"label": "FIN_PROCESS", "pattern": "погашение"},
    {"label": "FIN_PROCESS", "pattern": "начисление процентов"},
    {"label": "FIN_PROCESS", "pattern": "начисление"},
    {"label": "FIN_PROCESS", "pattern": "досрочное расторжение"},
    {"label": "FIN_PROCESS", "pattern": "расторжение договора"},
    {"label": "FIN_PROCESS", "pattern": "доверительное управление"},
    {"label": "FIN_PROCESS", "pattern": "листинг"},
    {"label": "FIN_PROCESS", "pattern": "размещение облигаций"},
    {"label": "FIN_PROCESS", "pattern": "конвертация"},
    {"label": "FIN_PROCESS", "pattern": "пролонгация"},
    {"label": "FIN_PROCESS", "pattern": "снятие наличных"},
    {"label": "FIN_PROCESS", "pattern": "капитализация процентов"},

    # ── Condition ─────────────────────────────────────────────────────────────
    {"label": "FIN_CONDITION", "pattern": "при досрочном погашении"},
    {"label": "FIN_CONDITION", "pattern": "при полном погашении задолженности"},
    {"label": "FIN_CONDITION", "pattern": "при утрате залога"},
    {"label": "FIN_CONDITION", "pattern": "не менее трёх лет"},

    # ── LegalTerm ─────────────────────────────────────────────────────────────
    {"label": "FIN_LEGAL", "pattern": "кредитный договор"},
    {"label": "FIN_LEGAL", "pattern": "договор"},
    {"label": "FIN_LEGAL", "pattern": "поручительство"},
    {"label": "FIN_LEGAL", "pattern": "залог"},
    {"label": "FIN_LEGAL", "pattern": "оферта"},
    {"label": "FIN_LEGAL", "pattern": "страховой случай"},
    {"label": "FIN_LEGAL", "pattern": "налоговый вычет"},
    {"label": "FIN_LEGAL", "pattern": "налоговый вычет типа А"},
    {"label": "FIN_LEGAL", "pattern": "НДФЛ"},
    {"label": "FIN_LEGAL", "pattern": "право собственности"},
    {"label": "FIN_LEGAL", "pattern": "основной долг"},
]


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS ДЛЯ РЕЗУЛЬТАТА
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    id:          str
    text:        str
    cls:         str        # класс онтологии
    spacy_label: str        # оригинальная метка spaCy
    start:       int        # символьная позиция
    end:         int
    confidence:  float = 0.8

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "text":       self.text,
            "class":      self.cls,
            "start":      self.start,
            "confidence": self.confidence,
            "comment":    "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# NER
# ─────────────────────────────────────────────────────────────────────────────

class SpacyFinancialNER:
    """
    Финансовый NER на базе spaCy.

    Архитектура двухуровневая:
      1. EntityRuler (before ner) — доменные паттерны, высокий приоритет
      2. spaCy NER — ловит то, что не покрыто словарём (организации, суммы)
    """

    def __init__(self, model: str = "ru_core_news_sm"):
        self.nlp = spacy.load(model)
        self._add_ruler()

    def _add_ruler(self):
        """Добавляем EntityRuler перед стандартным NER."""
        # "before ner" — ruler имеет приоритет над статистической моделью
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(RULER_PATTERNS)

    def extract(self, text: str) -> list[Entity]:
        doc = self.nlp(text)
        entities = []

        for i, ent in enumerate(doc.ents, start=1):
            spacy_label = ent.label_
            onto_cls = SPACY_TO_ONTOLOGY.get(spacy_label)

            # Пропускаем метки, которых нет в нашей схеме
            if onto_cls is None:
                continue

            entities.append(Entity(
                id=f"e{i}",
                text=ent.text,
                cls=onto_cls,
                spacy_label=spacy_label,
                start=ent.start_char,
                end=ent.end_char,
                # EntityRuler-сущности чуть более уверенные, чем статистические
                confidence=0.85 if spacy_label.startswith("FIN_") else 0.70,
            ))

        return entities
