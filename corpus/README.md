# Корпус финансовых документов
## Структура и пайплайн разметки

```
corpus/
├── documents.py          ← 16 документов корпуса
├── gold_standard.json    ← эталонная разметка (doc_001, doc_008)
├── auto_annotate.py      ← скрипт LLM-разметки
├── review_annotations.py ← инструмент ручной проверки
├── annotations/          ← создаётся автоматически
│   ├── doc_001_annotation.json
│   ├── doc_001_review.json   ← заполняешь вручную
│   └── all_annotations.json
└── README.md
```

---

## Шаг 1. Установка зависимостей

```bash
pip install openai python-dotenv
```

Создай файл `.env` в папке `corpus/`:

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
```

---

## Шаг 2. Запуск разметки

```bash
# Разметить весь корпус (16 документов)
python auto_annotate.py

# Разметить один документ
python auto_annotate.py --doc doc_001

# Разметить несколько документов
python auto_annotate.py --doc doc_001 doc_005 doc_008

# Использовать более дешёвую модель
python auto_annotate.py --model gpt-4o-mini
```

---

## Шаг 3. Проверка разметки вручную

```bash
# Просмотр всех размеченных документов
python review_annotations.py

# Просмотр одного документа
python review_annotations.py --doc doc_001

# Сравнение с gold standard (для doc_001 и doc_008)
python review_annotations.py --compare

# Один документ + сравнение
python review_annotations.py --doc doc_001 --compare
```

Скрипт создаст файлы `{doc_id}_review.json` — заполняй их вручную:

```json
{
  "entities_review": [
    {
      "id": "e1",
      "text": "ипотечный кредит",
      "llm_class": "FinancialProduct",
      "verdict": "ok",         ← ok / wrong_class / spurious
      "correct_class": null,
      "comment": ""
    }
  ],
  "missing_entities": [
    {"text": "...", "class": "...", "comment": "пропущено LLM"}
  ]
}
```

---

## Схема классов

| Класс | Пример |
|---|---|
| FinancialProduct | ипотечный кредит, ПИФ, облигация |
| ProductAttribute | процентная ставка, срок, лимит |
| Actor | банк, заёмщик, управляющая компания |
| Process | погашение, выдача, начисление |
| Condition | возраст от 21 года, срок владения 180 дней |
| LegalTerm | залог, оферта, поручительство |
| Metric | 10.5% годовых, 1000 рублей, 20 лет |

---

## Эталонные примеры разметки (gold standard)

Для двух документов уже сделана полная ручная разметка:

- **doc_001** — ипотечный кредит (14 сущностей, 10 отношений)
- **doc_008** — ПИФ (16 сущностей, 12 отношений)

Используй их как ориентир при проверке остальных.
