"""
pipeline.py
────────────
LangGraph пайплайн мультиагентной системы.

Граф:
    extract_terms → classify_entities → extract_relations → validate → integrate

Запуск:
    export OPENAI_API_KEY=sk-...
    python pipeline.py                       # весь корпус
    python pipeline.py --doc doc_001         # один документ
    python pipeline.py --doc doc_001 --eval  # + сравнение с gold standard

Зависимости:
    pip install langchain langchain-openai langgraph rdflib
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from agents.state import PipelineState
from agents.term_extractor import TermExtractorAgent
from agents.entity_classifier import EntityClassifierAgent
from agents.relation_extractor import RelationExtractorAgent
from agents.validator import ValidatorAgent
from agents.ontology_integrator import OntologyIntegratorAgent, build_base_graph
from evaluate import evaluate
from corpus.loader import load_corpus, load_gold

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ПОСТРОЕНИЕ ГРАФА LANGGRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    llm:          ChatOpenAI,
    integrator:   OntologyIntegratorAgent,
    corpus_texts: list[str],
) -> StateGraph:
    """
    Создаёт и компилирует LangGraph граф агентов.

    Узлы:
        extract_terms      → TermExtractorAgent
        classify_entities  → EntityClassifierAgent
        extract_relations  → RelationExtractorAgent
        validate           → ValidatorAgent
        integrate          → OntologyIntegratorAgent

    Рёбра: линейные (каждый агент передаёт состояние следующему).
    """

    term_agent  = TermExtractorAgent(llm=llm, corpus_texts=corpus_texts)
    class_agent = EntityClassifierAgent(llm=llm)
    rel_agent   = RelationExtractorAgent(llm=llm)
    val_agent   = ValidatorAgent(llm=llm)

    def node_extract(state):  return term_agent.run(state)
    def node_classify(state): return class_agent.run(state)
    def node_relate(state):   return rel_agent.run(state)
    def node_validate(state): return val_agent.run(state)
    def node_integrate(state):return integrator.run(state)

    graph = StateGraph(PipelineState)
    graph.add_node("extract_terms",     node_extract)
    graph.add_node("classify_entities", node_classify)
    graph.add_node("extract_relations", node_relate)
    graph.add_node("validate",          node_validate)
    graph.add_node("integrate",         node_integrate)

    graph.set_entry_point("extract_terms")
    graph.add_edge("extract_terms",     "classify_entities")
    graph.add_edge("classify_entities", "extract_relations")
    graph.add_edge("extract_relations", "validate")
    graph.add_edge("validate",          "integrate")
    graph.add_edge("integrate",         END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# ЗАПУСК ПО ОДНОМУ ДОКУМЕНТУ
# ─────────────────────────────────────────────────────────────────────────────

def run_doc(app, doc: dict, verbose: bool = True) -> PipelineState:
    """Прогоняет один документ через граф."""
    import time

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Документ: {doc['id']}  ({len(doc['text'])} символов)")
        print(f"{'─'*55}")

    initial = PipelineState(
        doc_id=doc["id"],
        doc_type=doc["type"],
        product=doc["product"],
        text=doc["text"],
    )

    t0    = time.time()
    raw   = app.invoke(initial)
    elapsed = time.time() - t0

    # LangGraph с dataclass-состоянием возвращает dict — конвертируем обратно
    if isinstance(raw, dict):
        final = PipelineState(**{
            k: v for k, v in raw.items()
            if k in PipelineState.__dataclass_fields__
        })
    else:
        final = raw

    if verbose:
        print(f"\n  Результат [{doc['id']}]")
        print(f"    терминов:   {len(final.terms)}")
        print(f"    сущностей:  {len(final.entities)}")
        print(f"    отношений:  {len(final.relations)}")
        print(f"    троек:      {len(final.validated_triples)}")
        print(f"    время:      {elapsed:.1f}с")
        if final.errors:
            for err in final.errors:
                print(f"    ⚠ {err}")

    return final


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Мультиагентная система построения онтологий (LangGraph)"
    )
    parser.add_argument("--doc",   nargs="*", help="ID документов (по умолчанию — все)")
    parser.add_argument("--eval",  action="store_true", help="Оценить по gold standard")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI модель")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Задай переменную OPENAI_API_KEY")

    verbose      = not args.quiet
    docs         = load_corpus(doc_ids=args.doc)
    corpus_texts = [d["text"] for d in load_corpus() if d["text"]]

    print(f"\n{'='*60}")
    print(f"  Мультиагентная система | LangGraph + {args.model}")
    print(f"  Документов: {len(docs)}")
    print(f"{'='*60}")

    llm        = ChatOpenAI(model=args.model, temperature=0, api_key=api_key)
    base_graph = build_base_graph()
    integrator = OntologyIntegratorAgent(graph=base_graph)
    app        = build_graph(llm=llm, integrator=integrator, corpus_texts=corpus_texts)

    results: list[PipelineState] = []
    for doc in docs:
        results.append(run_doc(app, doc, verbose=verbose))

    integrator.save(str(OUTPUT_DIR))

    # Сохраняем результаты в формате gold_annotations.json
    if len(results) == 1:
        ann_path = OUTPUT_DIR / f"{results[0].doc_id}.json"
    else:
        ann_path = OUTPUT_DIR / "results.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump({"documents": [r.to_dict() for r in results]},
                  f, ensure_ascii=False, indent=2)
    print(f"  Аннотации: {ann_path}")

    if args.eval:
        gold      = load_gold()
        gold_path = OUTPUT_DIR / "gold_annotations.json"
        with open(gold_path, "w", encoding="utf-8") as f:
            json.dump(gold, f, ensure_ascii=False, indent=2)

        m = evaluate(
            results=results,
            gold_path=gold_path,
            get_entities=lambda s: [
                {"text": e.text, "class": e.cls} for e in s.entities
            ],
            get_relations=lambda s: s.validated_triples,
            get_doc_id=lambda s: s.doc_id,
        )

        _print_metrics(m)

        eval_path = OUTPUT_DIR / "eval.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        print(f"  Метрики: {eval_path}")

    print(f"\n{'='*60}\n")


def _print_metrics(m: dict):
    print(f"\n{'─'*60}  ОЦЕНКА\n")

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
