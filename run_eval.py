"""Run the agent on eval/questions.json and score it.

    python run_eval.py run baseline
    python run_eval.py run more_chunks --k 8
    python run_eval.py run prompt_v2 --prompt v2
    python run_eval.py compare baseline more_chunks
"""
import argparse
import json
import re
import time
from pathlib import Path

QUESTIONS_FILE = Path("eval/questions.json")
RESULTS_DIR = Path("results")

REFUSAL_PHRASES = [
    "could not find", "couldn't find", "can't find", "cannot find", "unable to find",
    "not find", "not in these documents", "does not contain", "doesn't contain",
    "not contain", "no information", "not mentioned", "not provided", "not reported",
    "not available in", "do not have", "don't have",
]


def normalise(text):
    return re.sub(r"[\s,]", "", text.lower())


def matches_all(answer, groups):
    # every group needs at least one of its alternatives in the answer
    a = normalise(answer)
    return all(any(normalise(alt) in a for alt in group) for group in groups)


def is_refusal(answer):
    low = answer.lower()
    return any(phrase in low for phrase in REFUSAL_PHRASES)


def retrieved_pages(steps):
    seen = set()
    for step in steps:
        if step["tool"] != "search_filings":
            continue
        for hit in step["output"].get("results", []):
            m = re.match(r"(.+) p\.(\d+)$", hit["source"])
            if m:
                seen.add((m.group(1), int(m.group(2))))
    return seen


def score_question(q, out):
    expected_pages = q.get("pages")
    if expected_pages:
        seen = retrieved_pages(out["steps"])
        hit = any((doc, page) in seen for doc, pages in expected_pages.items() for page in pages)
    else:
        hit = None

    if q.get("should_refuse"):
        correct = is_refusal(out["answer"])
    else:
        correct = matches_all(out["answer"], q["expected"])

    used = {s["tool"] for s in out["steps"]}
    tools_ok = set(q.get("tools", [])) <= used
    return {"correct": correct, "retrieval_hit": hit, "tools_ok": tools_ok}


def run_questions(questions, run_one):
    results = []
    for i, q in enumerate(questions, start=1):
        out = run_one(q["question"])
        s = score_question(q, out)
        results.append({
            "id": q["id"], "type": q["type"], "question": q["question"], "answer": out["answer"],
            **s,
            "n_steps": len(out["steps"]),
            "tool_calls": [{"tool": st["tool"], "args": st["args"]} for st in out["steps"]],
            "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"],
            "stopped": out["stopped"],
        })
        print(f"[{i}/{len(questions)}] {q['id']}: {'PASS' if s['correct'] else 'FAIL'}"
              f" (retrieval {'hit' if s['retrieval_hit'] else 'miss' if s['retrieval_hit'] is False else 'n/a'})")
    return results


def summarise(results):
    n = len(results)
    correct = sum(r["correct"] for r in results)
    applicable = [r for r in results if r["retrieval_hit"] is not None]
    refusals = [r for r in results if r["type"] == "refusal"]
    tokens = sum(r["input_tokens"] + r["output_tokens"] for r in results)
    failed = [r for r in results if not r["correct"]]
    return {
        "questions": n,
        "answer_accuracy": correct / n if n else 0,
        "retrieval_hit_rate": (sum(r["retrieval_hit"] for r in applicable) / len(applicable)) if applicable else None,
        "refusal_accuracy": (sum(r["correct"] for r in refusals) / len(refusals)) if refusals else None,
        "tools_ok_rate": sum(r["tools_ok"] for r in results) / n if n else 0,
        "avg_steps": sum(r["n_steps"] for r in results) / n if n else 0,
        "avg_tokens": tokens / n if n else 0,
        "tokens_per_correct_answer": tokens / correct if correct else None,
        "fail_retrieval_miss": sum(1 for r in failed if r["retrieval_hit"] is False),
        "fail_after_retrieval_hit": sum(1 for r in failed if r["retrieval_hit"] is True),
        "fail_refusal_or_other": sum(1 for r in failed if r["retrieval_hit"] is None),
    }


def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def print_report(label, results):
    s = summarise(results)
    print(f"\n=== {label} ===")
    print(f"answer accuracy:     {pct(s['answer_accuracy'])}  ({round(s['answer_accuracy'] * s['questions'])}/{s['questions']})")
    print(f"retrieval hit rate:  {pct(s['retrieval_hit_rate'])}")
    print(f"refusal accuracy:    {pct(s['refusal_accuracy'])}")
    print(f"right tools used:    {pct(s['tools_ok_rate'])}")
    print(f"avg steps / tokens:  {s['avg_steps']:.1f} / {s['avg_tokens']:.0f}")
    if s["tokens_per_correct_answer"]:
        print(f"tokens per correct answer: {s['tokens_per_correct_answer']:.0f}")
    print(f"failures: {s['fail_retrieval_miss']} retrieval miss, "
          f"{s['fail_after_retrieval_hit']} wrong even though retrieval hit, "
          f"{s['fail_refusal_or_other']} refusal/other")
    for r in results:
        if not r["correct"]:
            print(f"  FAIL {r['id']}: {r['answer'][:140]!r}")


def compare(a, b):
    by_id = {r["id"]: r for r in a["results"]}
    fixed, broken = [], []
    for r in b["results"]:
        before = by_id.get(r["id"])
        if before is None:
            continue
        if not before["correct"] and r["correct"]:
            fixed.append(r["id"])
        elif before["correct"] and not r["correct"]:
            broken.append(r["id"])
    sa, sb = summarise(a["results"]), summarise(b["results"])
    print(f"\n{a['label']}  ->  {b['label']}")
    for key in ("answer_accuracy", "retrieval_hit_rate", "refusal_accuracy"):
        print(f"  {key:20} {pct(sa[key]):>5} -> {pct(sb[key]):>5}")
    print(f"  {'avg_tokens':20} {sa['avg_tokens']:>5.0f} -> {sb['avg_tokens']:>5.0f}")
    print(f"  fixed:  {fixed or 'none'}")
    print(f"  broken: {broken or 'none'}")
    return {"fixed": fixed, "broken": broken}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("label")
    r.add_argument("--k", type=int, help="chunks returned per search (default: agent.SEARCH_K)")
    r.add_argument("--model", help="model name (default: agent.MODEL)")
    r.add_argument("--prompt", default="v1", help="system prompt version: v1 (baseline) or v2")
    r.add_argument("--questions", default=str(QUESTIONS_FILE))
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    args = ap.parse_args()

    if args.cmd == "compare":
        load = lambda name: json.loads((RESULTS_DIR / f"{name}.json").read_text())
        compare(load(args.before), load(args.after))
        return

    from dotenv import load_dotenv
    import anthropic
    import agent
    import rag

    load_dotenv()
    if args.k:
        agent.SEARCH_K = args.k
    model = args.model or agent.MODEL
    client, index = anthropic.Anthropic(), rag.load_index()
    questions = json.loads(Path(args.questions).read_text())

    results = run_questions(questions, lambda text: agent.run_agent(text, client, index, model=model,
                                                                 prompt_version=args.prompt))

    RESULTS_DIR.mkdir(exist_ok=True)
    record = {"label": args.label, "model": model, "k": agent.SEARCH_K, "prompt": args.prompt,
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}
    (RESULTS_DIR / f"{args.label}.json").write_text(json.dumps(record, indent=2))
    print_report(args.label, results)


if __name__ == "__main__":
    main()