# Filings research agent

Asks questions about two company quarterly reports (Agnico Eagle, Evolution Mining)
and answers with the page number.

It has two parts: search over the PDFs (chunks + embeddings), and an agent that
can call a search tool and a calculator. It stops after 8 steps or 60k tokens.

## Run it

```
pip install -r requirements.txt
```

Put your key in `.env` as `ANTHROPIC_API_KEY=...` and the PDFs in `docs/` (I have included the PDFs I tested on ), then:

```
python rag.py build
python agent.py "What was Agnico Eagle's realized gold price in Q3 2026?"
python run_eval.py run baseline
```

## How I tested it

10 questions in `eval/questions.json`: 8 lookups/calculations and 2 that the reports
can't answer (the agent should say it can't find them). I wrote the expected answers
before running anything. Results are in `results/`.

| run                                       | correct      |
| ----------------------------------------- | ------------ |
| baseline (5 chunks per search), run twice | 9/10, 9/10   |
| 8 chunks per search, run twice            | 10/10, 10/10 |
| extra rule in the prompt                  | 9/10         |

Only one question ever changed: the total cash cost per ounce. The report has a
company-wide figure ($1,133) and a single-mine figure ($1,177) that look alike. Baseline
once gave the mine figure and once ran out of tokens searching. 8 chunks got it right both
times, my prompt fix didn't help.

## Rough edges

Only 10 questions, one report. Scoring is string matching. More questions coming.
