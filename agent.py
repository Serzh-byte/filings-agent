"""Tool-calling agent over the filings index.

    python agent.py "What was Agnico Eagle's payable gold production in Q3 2026?"
"""
import json
import os
import sys

import rag

MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-5-5")
MAX_STEPS = 8
MAX_TOTAL_TOKENS = 60_000
SEARCH_K = 5


def search_filings(query, company=None, *, index, embed_fn=rag.embed, k=None):
    k = k or SEARCH_K  # looked up at call time so run_eval can override it
    vectors, records = index
    known = sorted({r["company"] for r in records})
    if company and company.lower() not in [c.lower() for c in known]:
        return {"error": f"No documents for company '{company}'. Available companies: {known}"}
    hits = rag.search(query, vectors, records, embed_fn=embed_fn, k=k, company=company)
    return {
        "results": [
            {"source": f"{h['doc']} p.{h['page']}", "company": h["company"], "text": h["text"]}
            for h in hits
        ]
    }


def calculate(operation, a, b):
    try:
        if operation == "percent_change":
            result = (b - a) / a * 100
        elif operation == "difference":
            result = b - a
        elif operation == "ratio":
            result = a / b
        else:
            return {"error": f"Unknown operation '{operation}'. Use percent_change, difference or ratio."}
    except ZeroDivisionError:
        return {"error": "Division by zero."}
    return {"operation": operation, "a": a, "b": b, "result": round(result, 4)}


TOOLS = [
    {
        "name": "search_filings",
        "description": (
            "Search the text of the company reports. Returns the most relevant passages, "
            "each with its source document and page. Use it to find facts and figures."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in plain words."},
                "company": {
                    "type": "string",
                    "description": "Company code to restrict the search to. Use it when the question names one company.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Exact arithmetic on two numbers. percent_change = (b - a) / a * 100, "
            "difference = b - a, ratio = a / b. Use it for any calculation instead of doing maths yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["percent_change", "difference", "ratio"]},
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
]


# v2 = v1 plus this one rule
EXTRA_RULES_V2 = """
- Read the row and column labels of every table before quoting a number. Unless the question names a specific mine or site, use the company-wide (consolidated) figure, never the figure for a single mine."""


def build_system_prompt(records, version="v1"):
    companies = sorted({r["company"] for r in records})
    prompt = f"""You answer questions about company quarterly reports using ONLY the search tool.

Companies available: {companies}.

Rules:
- Find facts with search_filings. Do not answer from memory.
- Cite the source (document and page) for every figure you state.
- Always state units and currency. Never treat figures in different currencies as directly comparable.
- Use the calculate tool for any arithmetic.
- If the reports do not contain the answer, say exactly that you could not find it in these documents. Do not guess.
- Be concise."""
    if version == "v2":
        prompt += EXTRA_RULES_V2
    return prompt


def run_tool(name, args, index, embed_fn=rag.embed):
    try:
        if name == "search_filings":
            return search_filings(index=index, embed_fn=embed_fn, **args)
        if name == "calculate":
            return calculate(**args)
        return {"error": f"Unknown tool '{name}'."}
    except Exception as exc:  # hand the error to the model instead of crashing
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_agent(question, client, index, model=MODEL, max_steps=MAX_STEPS,
              max_total_tokens=MAX_TOTAL_TOKENS, embed_fn=rag.embed, prompt_version="v1"):
    system = build_system_prompt(index[1], prompt_version)
    messages = [{"role": "user", "content": question}]
    steps, seen_calls = [], set()
    in_tokens = out_tokens = 0

    def result(answer, stopped):
        return {"question": question, "answer": answer, "stopped": stopped, "steps": steps,
                "input_tokens": in_tokens, "output_tokens": out_tokens}

    for _ in range(max_steps):
        response = client.messages.create(
            model=model, max_tokens=1024, system=system, tools=TOOLS, messages=messages
        )
        in_tokens += response.usage.input_tokens
        out_tokens += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            return result(text, "answered")

        if in_tokens + out_tokens > max_total_tokens:
            return result("Stopped: token budget exceeded.", "budget")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            key = (block.name, json.dumps(block.input, sort_keys=True))
            if key in seen_calls:
                output = {"error": "You already made this exact call. Use the earlier result, "
                                   "or answer with what you have."}
            else:
                seen_calls.add(key)
                output = run_tool(block.name, block.input, index, embed_fn)
            steps.append({"tool": block.name, "args": block.input, "output": output})
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)}
            )
        messages.append({"role": "user", "content": tool_results})

    return result("Stopped: reached the step limit without a final answer.", "max_steps")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    from dotenv import load_dotenv
    import anthropic

    load_dotenv()
    out = run_agent(sys.argv[1], anthropic.Anthropic(), rag.load_index())

    for i, step in enumerate(out["steps"], start=1):
        shown = json.dumps(step["output"])[:150]
        print(f"step {i}: {step['tool']}({json.dumps(step['args'])}) -> {shown}...")
    print(f"\nANSWER ({out['stopped']}): {out['answer']}")
    print(f"tokens: {out['input_tokens']} in, {out['output_tokens']} out")