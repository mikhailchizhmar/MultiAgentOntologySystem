"""
evaluate.py
────────────
Единый модуль оценки качества для baseline и мультиагентной системы.

Три уровня оценки:
  1. Entity partial match  — сущности с учётом нечёткого совпадения текста
  2. Entity + class match  — то же, но TP только при совпадении и текста, и класса
  3. Relation triple match — тройка (subject, relation, object) целиком
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# КОНСТАНТЫ
# ─────────────────────────────────────────────────────────────────────────────

# Порог сходства для partial match.
# Ниже — считаем FP/FN, выше — TP.
# 0.85 подобран эмпирически: отсекает реально разные сущности («банк» vs «банк России»),
# но принимает варианты написания («заёмщик» vs «заемщик», «ПСК» vs «пск»).
PARTIAL_MATCH_THRESHOLD = 0.85

# Словарь аббревиатур → полная форма.
# SequenceMatcher работает на символьном уровне и не знает, что ПСК = полная стоимость кредита.
# Нормализуем до сравнения, чтобы покрыть этот случай.
SYNONYMS: dict[str, str] = {
    "пск":  "полная стоимость кредита",
    "иис":  "индивидуальный инвестиционный счёт",
    "пиф":  "паевой инвестиционный фонд",
    "сча":  "стоимость чистых активов",
    "ук":   "управляющая компания",
    "асв":  "агентство по страхованию вкладов",
    "ндфл": "налог на доходы физических лиц",
}

# Все классы онтологии — для разбивки F1 по классам
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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Нижний регистр + замена аббревиатур через словарь синонимов."""
    t = text.lower().strip()
    return SYNONYMS.get(t, t)


def similarity(a: str, b: str) -> float:
    """Символьное сходство двух строк через SequenceMatcher (0.0 – 1.0)."""
    return SequenceMatcher(None, a, b).ratio()


def find_best_match(
    pred: str,
    gold_list: list[str],
    used: set[int],
) -> tuple[int | None, float]:
    """
    Для предсказанной сущности ищет наиболее похожую незанятую gold-сущность.
    Возвращает (индекс, score) или (None, 0.0) если ничего не нашлось.

    used — множество индексов gold-сущностей, уже засчитанных как TP.
    Каждая gold-сущность может быть использована только один раз,
    иначе одна хорошая предсказанная сущность «закрыла» бы несколько gold.
    """
    best_score = 0.0
    best_idx   = None
    pred_norm  = normalize(pred)

    for idx, gold in enumerate(gold_list):
        if idx in used:
            continue
        score = similarity(pred_norm, normalize(gold))
        if score > best_score:
            best_score = score
            best_idx   = idx

    return best_idx, best_score


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Возвращает (precision, recall, f1) из TP/FP/FN."""
    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 3), round(r, 3), round(f1, 3)


# ─────────────────────────────────────────────────────────────────────────────
# PARTIAL MATCH ДЛЯ СУЩНОСТЕЙ (текст без учёта класса)
# ─────────────────────────────────────────────────────────────────────────────

def entity_partial_match(
    pred_entities: list[dict],
    gold_entities: list[dict],
) -> dict:
    """
    Считает TP/FP/FN по тексту сущности через нечёткое совпадение.

    Класс при этом НЕ учитывается — только факт нахождения текста.
    Это базовая метрика: «система вообще нашла нужные термины?»
    """
    gold_texts = [e["text"] for e in gold_entities]
    pred_texts = [e["text"] for e in pred_entities]

    used_gold: set[int] = set()
    tp = fp = 0

    for pred in pred_texts:
        idx, score = find_best_match(pred, gold_texts, used_gold)
        if idx is not None and score >= PARTIAL_MATCH_THRESHOLD:
            tp += 1
            used_gold.add(idx)
        else:
            fp += 1

    fn = len(gold_texts) - len(used_gold)
    p, r, f1 = prf(tp, fp, fn)

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY + CLASS MATCH (текст И класс должны совпасть)
# ─────────────────────────────────────────────────────────────────────────────

def entity_class_match(
    pred_entities: list[dict],
    gold_entities: list[dict],
) -> dict:
    """
    TP только если совпадают и текст (partial match), и класс сущности.

    Это строгая метрика: «система не только нашла термин, но и правильно его классифицировала».
    Для онтологии важнее именно эта метрика — неверный класс ломает граф.

    Дополнительно считает F1 отдельно по каждому классу (per-class breakdown),
    чтобы видеть где система ошибается систематически.
    """
    gold_texts   = [e["text"]  for e in gold_entities]
    gold_classes = [e["class"] for e in gold_entities]
    pred_texts   = [e["text"]  for e in pred_entities]
    pred_classes = [e["class"] for e in pred_entities]

    used_gold: set[int] = set()
    tp = fp = 0
    class_errors = []

    # Счётчики per-class: для каждого класса считаем TP/FP/FN отдельно
    per_class: dict[str, dict] = {
        cls: {"tp": 0, "fp": 0, "fn": 0} for cls in ENTITY_CLASSES
    }

    for i, pred_text in enumerate(pred_texts):
        pred_cls = pred_classes[i]
        idx, score = find_best_match(pred_text, gold_texts, used_gold)

        if idx is not None and score >= PARTIAL_MATCH_THRESHOLD:
            gold_cls = gold_classes[idx]

            if pred_cls == gold_cls:
                # Текст совпал И класс совпал → TP
                tp += 1
                used_gold.add(idx)
                if pred_cls in per_class:
                    per_class[pred_cls]["tp"] += 1
            else:
                # Текст совпал, но класс неверный → FP для pred_cls, FN для gold_cls
                fp += 1
                class_errors.append({
                    "text":      pred_text,
                    "predicted": pred_cls,
                    "gold":      gold_cls,
                })
                if pred_cls in per_class:
                    per_class[pred_cls]["fp"] += 1
                if gold_cls in per_class:
                    per_class[gold_cls]["fn"] += 1
        else:
            # Текст не найден в gold вообще → FP
            fp += 1
            if pred_cls in per_class:
                per_class[pred_cls]["fp"] += 1

    # FN: gold-сущности, которые не нашла ни одна предсказанная
    fn = len(gold_texts) - len(used_gold)
    for idx, gold_cls in enumerate(gold_classes):
        if idx not in used_gold and gold_cls in per_class:
            per_class[gold_cls]["fn"] += 1

    p, r, f1 = prf(tp, fp, fn)

    # Считаем F1 по каждому классу
    per_class_f1 = {}
    for cls, counts in per_class.items():
        cp, cr, cf1 = prf(counts["tp"], counts["fp"], counts["fn"])
        per_class_f1[cls] = {"tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
                              "precision": cp, "recall": cr, "f1": cf1}

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": f1,
        "class_errors":  class_errors,
        "per_class_f1":  per_class_f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RELATION TRIPLE MATCH
# ─────────────────────────────────────────────────────────────────────────────

def relation_triple_match(
    pred_relations: list[dict],
    gold_relations: list[dict],
    gold_entities:  list[dict],
) -> dict:
    """
    Считает TP/FP/FN для отношений.

    Тройка (subject_text, relation, object_text) засчитывается как TP
    только если совпадают все три компонента.

    Subject и object сравниваются через partial match по тексту —
    потому что предсказанный текст сущности может незначительно отличаться от gold.
    Тип отношения (relation) должен совпасть точно.

    Это ключевая метрика для сравнения baseline vs мультиагентная система:
    baseline строит отношения по шаблонным правилам и ошибается на большинстве пар,
    LLM-агент опирается на семантику текста.
    """
    # Строим индекс: id → text для gold-сущностей (нужен для восстановления текста из id)
    gold_ent_by_id = {e["id"]: e["text"] for e in gold_entities}

    # Нормализуем gold-тройки в формат (subj_text, relation, obj_text)
    gold_triples = []
    for r in gold_relations:
        subj = gold_ent_by_id.get(r["subject"], r.get("subject_text", ""))
        obj  = gold_ent_by_id.get(r["object"],  r.get("object_text",  ""))
        if subj and obj:
            gold_triples.append((subj, r["relation"], obj))

    # Нормализуем предсказанные тройки
    pred_triples = []
    for r in pred_relations:
        subj = r.get("subject_text", "")
        obj  = r.get("object_text",  "")
        rel  = r.get("relation",     "")
        if subj and obj and rel:
            pred_triples.append((subj, rel, obj))

    used_gold: set[int] = set()
    tp = fp = 0

    for pred_subj, pred_rel, pred_obj in pred_triples:
        matched = False

        for idx, (gold_subj, gold_rel, gold_obj) in enumerate(gold_triples):
            if idx in used_gold:
                continue

            # Relation должен совпасть точно — это контролируемый словарь
            if pred_rel != gold_rel:
                continue

            # Subject и object сравниваем через partial match
            subj_score = similarity(normalize(pred_subj), normalize(gold_subj))
            obj_score  = similarity(normalize(pred_obj),  normalize(gold_obj))

            if subj_score >= PARTIAL_MATCH_THRESHOLD and obj_score >= PARTIAL_MATCH_THRESHOLD:
                tp += 1
                used_gold.add(idx)
                matched = True
                break

        if not matched:
            fp += 1

    fn = len(gold_triples) - len(used_gold)
    p, r, f1 = prf(tp, fp, fn)

    # Разбивка FP по типам отношений — показывает какие отношения система «выдумывает»
    fp_by_relation: dict[str, int] = {}
    for pred_subj, pred_rel, pred_obj in pred_triples:
        found = any(
            pred_rel == gold_rel
            and similarity(normalize(pred_subj), normalize(gold_subj)) >= PARTIAL_MATCH_THRESHOLD
            and similarity(normalize(pred_obj),  normalize(gold_obj))  >= PARTIAL_MATCH_THRESHOLD
            for gold_subj, gold_rel, gold_obj in gold_triples
        )
        if not found:
            fp_by_relation[pred_rel] = fp_by_relation.get(pred_rel, 0) + 1

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": f1,
        "fp_by_relation": fp_by_relation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    results:   list,          # list[ProcessedDoc] или list[PipelineState]
    gold_path: Path,
    get_entities:  callable,  # лямбда: result → list[dict] с полями text, class
    get_relations: callable,  # лямбда: result → list[dict] с полями subject_text, relation, object_text
    get_doc_id:    callable,  # лямбда: result → str
) -> dict:
    """
    Универсальная функция оценки — работает и для baseline, и для агентной системы.

    Принимает лямбды-адаптеры, потому что ProcessedDoc и PipelineState
    хранят сущности в разных форматах.

    Возвращает словарь с тремя блоками метрик:
      - entity_partial:  F1 по тексту (без учёта класса)
      - entity_strict:   F1 по тексту + классу + разбивка по классам
      - relation:        F1 по тройкам (subj, rel, obj)
    """
    with open(gold_path, encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_by_id = {d["id"]: d for d in gold_data["documents"]}

    per_doc: dict[str, dict] = {}

    # Аккумуляторы для micro-average по всем документам
    acc: dict[str, dict[str, int]] = {
        "partial":  {"tp": 0, "fp": 0, "fn": 0},
        "strict":   {"tp": 0, "fp": 0, "fn": 0},
        "relation": {"tp": 0, "fp": 0, "fn": 0},
    }

    for result in results:
        doc_id = get_doc_id(result)
        if doc_id not in gold_by_id:
            continue

        gold_doc  = gold_by_id[doc_id]
        gold_ents = gold_doc["entities"]
        gold_rels = gold_doc["relations"]

        pred_ents = get_entities(result)
        pred_rels = get_relations(result)

        m_partial = entity_partial_match(pred_ents, gold_ents)
        m_strict  = entity_class_match(pred_ents, gold_ents)
        m_rel     = relation_triple_match(pred_rels, gold_rels, gold_ents)

        per_doc[doc_id] = {
            "entity_partial": m_partial,
            "entity_strict":  m_strict,
            "relation":       m_rel,
        }

        for key, m in [("partial", m_partial), ("strict", m_strict), ("relation", m_rel)]:
            acc[key]["tp"] += m["tp"]
            acc[key]["fp"] += m["fp"]
            acc[key]["fn"] += m["fn"]

    # Micro-average по всем документам
    def micro(key: str) -> dict:
        tp, fp, fn = acc[key]["tp"], acc[key]["fp"], acc[key]["fn"]
        p, r, f1 = prf(tp, fp, fn)
        return {"tp": tp, "fp": fp, "fn": fn,
                "micro_precision": p, "micro_recall": r, "micro_f1": f1}

    return {
        "per_doc": per_doc,
        "overall": {
            "entity_partial": micro("partial"),
            "entity_strict":  micro("strict"),
            "relation":       micro("relation"),
        },
    }
