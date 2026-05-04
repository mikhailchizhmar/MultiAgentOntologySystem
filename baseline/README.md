# Baseline: spaCy NER → rdflib OWL

Baseline 1 из сравнительной таблицы диссертации.
Никакого ML/LLM — только правила и словари.

## Структура

```
baseline/
├── baseline_ner.py       ← spaCy + EntityRuler
├── ontology_graph.py     ← rdflib граф
├── baseline_pipeline.py  ← главный скрипт
├── output/               ← создаётся при запуске
│   ├── baseline_ontology.ttl    ← открыть в Protégé
│   ├── baseline_ontology.owl
│   ├── baseline_annotations.json
│   └── baseline_eval.json       ← метрики (при --eval)
└── README.md
```

## Установка

```bash
pip install spacy rdflib
python -m spacy download ru_core_news_sm
```

## Запуск

```bash
# Весь корпус
python baseline_pipeline.py

# Один документ
python baseline_pipeline.py --doc doc_001

# С оценкой по gold standard
python baseline_pipeline.py --eval

# Более тяжёлая модель (лучше NER из коробки)
python baseline_pipeline.py --model ru_core_news_lg --eval
```

## Архитектура NER

```
Текст
  ↓
EntityRuler  ←── RULER_PATTERNS (словари доменных терминов)
  ↓
spaCy NER    ←── ru_core_news_sm (ORG, MONEY, DATE, ...)
  ↓
маппинг меток → классы онтологии
  ↓
list[Entity]
```

EntityRuler стоит **before ner** — доменные паттерны имеют приоритет
над статистической моделью. Это важно: `управляющая компания` должна
получить `Actor`, а не `ORG` (которое spaCy тоже маппит в `Actor`,
но с меньшей уверенностью и без гарантий).

## Архитектура извлечения отношений

Два прохода по парам сущностей в одном предложении:

1. **Текстовые паттерны** между сущностями  
   (`предоставляет` → `issuedBy`, `обязан` → `requires`, ...)

2. **Матрица типов** как fallback  
   `(FinancialProduct, ProductAttribute)` → `hasAttribute`

## Ожидаемые метрики (ориентир)

| Метрика   | Значение |
|-----------|----------|
| Precision | ~0.72    |
| Recall    | ~0.68    |
| F1        | ~0.70    |

Это и есть планка, которую мультиагентная LLM-система должна побить.
Ожидаем F1 ≥ 0.85 у агентной системы.
