"""
pipeline.py
────────────
LangGraph пайплайн мультиагентной системы.

Граф:
    extract_terms → classify_entities → extract_relations → validate → integrate

Запуск:
    export OPENAI_API_KEY=sk-...
    python pipeline.py                    # весь корпус
    python pipeline.py --doc doc_001      # один документ
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

# sys.path.insert(0, str(Path(__file__).parent.parent / "corpus"))

from corpus.documents import CORPUS
from agents.state import PipelineState
from agents.term_extractor import TermExtractorAgent
from agents.entity_classifier import EntityClassifierAgent
from agents.relation_extractor import RelationExtractorAgent
from agents.validator import ValidatorAgent
from agents.ontology_integrator import OntologyIntegratorAgent, build_base_graph

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

    term_agent     = TermExtractorAgent(llm=llm, corpus_texts=corpus_texts)
    class_agent    = EntityClassifierAgent(llm=llm)
    rel_agent      = RelationExtractorAgent(llm=llm)
    val_agent      = ValidatorAgent(llm=llm)

    # LangGraph требует, чтобы узлы возвращали dict или объект с __dict__
    # PipelineState — dataclass, поэтому оборачиваем в лямбды
    def node_extract(state: PipelineState)  -> PipelineState: return term_agent.run(state)
    def node_classify(state: PipelineState) -> PipelineState: return class_agent.run(state)
    def node_relate(state: PipelineState)   -> PipelineState: return rel_agent.run(state)
    def node_validate(state: PipelineState) -> PipelineState: return val_agent.run(state)
    def node_integrate(state: PipelineState)-> PipelineState: return integrator.run(state)

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
    initial = PipelineState(
        doc_id=doc["id"],
        doc_type=doc["type"],
        product=doc["product"],
        text=doc["text"],
    )

    final: PipelineState = app.invoke(initial)

    if verbose:
        print(f"\n  [{doc['id']}] {doc['product']}")
        print(f"    терминов:   {len(final["terms"])}")
        print(f"    сущностей:  {len(final["entities"])}")
        print(f"    отношений:  {len(final["relations"])}")
        print(f"    троек:      {len(final["validated_triples"])}")
        if final["errors"]:
            for err in final["errors"]:
                print(f"    ⚠ {err}")

    return final


# ─────────────────────────────────────────────────────────────────────────────
# ОЦЕНКА ПО GOLD STANDARD
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(results: list[PipelineState], gold_path: Path) -> dict:
    """Precision / Recall / F1 по сущностям относительно gold standard."""
    with open(gold_path, encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_by_id = {d["id"]: d for d in gold_data["documents"]}
    per_doc    = {}
    total_tp = total_fp = total_fn = 0

    for state in results:
        if state["doc_id"] not in gold_by_id:
            continue

        gold_ents = {e["text"].lower(): e["class"]
                     for e in gold_by_id[state["doc_id"]]["entities"]}
        pred_ents = {e.text.lower(): e.cls
                     for e in state["entities"]}

        gold_set, pred_set = set(gold_ents), set(pred_ents)
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        total_tp += tp; total_fp += fp; total_fn += fn

        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0

        per_doc[state["doc_id"]] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 3),
            "recall":    round(r, 3),
            "f1":        round(f1, 3),
            "missed":    sorted(gold_set - pred_set),
            "spurious":  sorted(pred_set - gold_set),
            "class_errors": [
                {"text": t, "gold": gold_ents[t], "pred": pred_ents[t]}
                for t in (gold_set & pred_set)
                if gold_ents[t] != pred_ents[t]
            ],
        }

    p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    return {
        "per_doc": per_doc,
        "overall": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "micro_precision": round(p, 3),
            "micro_recall":    round(r, 3),
            "micro_f1":        round(f1, 3),
        },
    }


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

    verbose = not args.quiet
    docs    = [d for d in CORPUS if d["id"] in args.doc] if args.doc else CORPUS

    print(f"\n{'='*60}")
    print(f"  Мультиагентная система | LangGraph + {args.model}")
    print(f"  Документов: {len(docs)}")
    print(f"{'='*60}")

    # Инициализация
    llm = ChatOpenAI(
        model=args.model,
        temperature=0,
        api_key=api_key,
    )

    base_graph  = build_base_graph()
    integrator  = OntologyIntegratorAgent(graph=base_graph)
    corpus_texts = [d["text"] for d in CORPUS]

    app = build_graph(llm=llm, integrator=integrator, corpus_texts=corpus_texts)

    # Обработка документов
    results: list[PipelineState] = []
    for doc in docs:
        state = run_doc(app, doc, verbose=verbose)
        results.append(state)

    # Сохранение онтологии
    integrator.save(str(OUTPUT_DIR))

    # Сохранение аннотаций
    # ann_path = OUTPUT_DIR / "annotations.json"
    # with open(ann_path, "w", encoding="utf-8") as f:
    #     json.dump(
    #         {
    #             "meta":      {"model": args.model, "docs": len(results)},
    #             "documents": [s for s in results],
    #         },
    #         f, ensure_ascii=False, indent=2,
    #     )
    # print(f"\n  Аннотации: {ann_path}")

    # Оценка
    if args.eval:
        gold_path = Path(__file__).parent.parent / "corpus" / "gold_annotations.json"
        if not gold_path.exists():
            print(f"  Gold standard не найден: {gold_path}")
        else:
            m = evaluate(results, gold_path)
            print(f"\n{'─'*60}  ОЦЕНКА")
            for doc_id, dm in m["per_doc"].items():
                print(f"\n  [{doc_id}]  "
                      f"P={dm['precision']}  R={dm['recall']}  F1={dm['f1']}  "
                      f"(TP={dm['tp']} FP={dm['fp']} FN={dm['fn']})")
                if dm["missed"]:
                    print(f"    пропущено: {dm['missed']}")
                if dm["class_errors"]:
                    print(f"    ошибки класса: {dm['class_errors']}")

            ov = m["overall"]
            print(f"\n  ИТОГО  "
                  f"P={ov['micro_precision']}  "
                  f"R={ov['micro_recall']}  "
                  f"F1={ov['micro_f1']}\n")

            eval_path = OUTPUT_DIR / "eval.json"
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=2)
            print(f"  Метрики: {eval_path}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
