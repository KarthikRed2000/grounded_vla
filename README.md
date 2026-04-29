# Grounded Vision-Language Agents for Instruction Following

CS 639: Intro to Foundation Models — Spring 2026 project.

This repo is the code companion to the proposal *Grounded Vision-Language
Agents for Instruction Following*. It evaluates three agents against three
benchmarks to test whether **visual grounding** and a **structured
Observe → Reason → Act (ORA) loop** improve multi-step instruction
following over text-only ReAct.

| Agent                   | Backbone            | Vision? | Agentic loop? |
|-------------------------|---------------------|---------|---------------|
| `react`                 | Mistral-7B-Instruct | no      | yes (ReAct)   |
| `single_shot`           | LLaVA-1.6-7B        | yes     | no            |
| `ora` (our method)      | LLaVA-1.6-7B        | yes     | yes (ORA)     |

The ORA loop's novelty — and the reason it's not just ReAct-with-images —
is that **the screenshot / diagram is re-encoded through the vision tower
at every step**, so the action decision is always grounded in the current
visual state, not a stale text summary.

## Quick start (no GPU required)

```bash
pip install -e .
grounded-vla smoke
```

`smoke` runs the full pipeline (ORA agent + mock backend) against three
bundled sample tasks. It exits nonzero if any piece of the pipeline
breaks — handy as a CI gate.

## GPU quick start

```bash
pip install -e .[gpu,data]
# one-time preprocessing of the public benchmarks
python scripts/prepare_mind2web.py --split test_task --out-dir data/mind2web --limit 200
python scripts/prepare_scienceqa.py --split test --out-dir data/scienceqa --limit 200

# run the primary method
grounded-vla eval \
  --config configs/ora_llava.yaml \
  --dataset-config configs/datasets/scienceqa.yaml \
  --limit 50 --save-dir runs/ora_scienceqa
```

`runs/ora_scienceqa/summary.json` holds the aggregate metrics; individual
trajectories land in `runs/ora_scienceqa/trajectories/` for error
analysis.

Run the full agent × dataset sweep with `python scripts/run_full_eval.py
--save-root runs/$(date +%F)`.

## Repo tour

```
grounded_vla/
├── schemas.py          # Task, Action, Observation, Trajectory, RunResult
├── action_parser.py    # Robust Thought/Action parser (JSON + NL fallback)
├── env.py              # Environments: StaticQAEnv and TaskReplayEnv
├── backends/
│   ├── base.py         # Backend interface
│   ├── mock.py         # Deterministic stand-in (used by CI + smoke)
│   ├── llava.py        # LLaVA-1.6-7B (lazy-loaded, 4-bit quantized)
│   └── mistral.py      # Mistral-7B-Instruct (text-only baseline)
├── agents/
│   ├── base.py         # Agent base class
│   ├── prompts.py      # ReAct / single-shot / ORA prompt templates
│   ├── react_agent.py  # Baseline 1
│   ├── single_shot_agent.py   # Baseline 2
│   └── ora_agent.py    # Our method (H2)
├── data/
│   ├── base.py         # Streaming Dataset + JsonlDataset
│   ├── mind2web.py     # Mind2Web loader (HF + JSONL modes)
│   ├── scienceqa.py    # ScienceQA loader
│   └── synthetic.py    # Synthetic corpus loader
├── synthetic/
│   ├── builder.py      # Generates candidate triples from a CC manifest
│   └── review.py       # Two-person review queue (Section 3.3)
├── eval/
│   ├── metrics.py      # Per-benchmark success + step efficiency
│   ├── error_analysis.py  # visual_misgrounding / reasoning / parse / truncated
│   └── runner.py       # Ties it all together; writes runs/ artifacts
├── lora.py             # Stretch goal (H3): PEFT-LoRA fine-tuning on synthetic
├── cli.py              # `grounded-vla` entry point
└── utils/              # Image loader + Rich-flavored logger
```

## Map from proposal → code

| Proposal section                                   | Code                                                      |
|----------------------------------------------------|-----------------------------------------------------------|
| §3.1 Backbone VLM (LLaVA-1.6)                      | `grounded_vla/backends/llava.py`                          |
| §3.1 Text-only baseline (Mistral-7B + ReAct)       | `grounded_vla/backends/mistral.py`, `agents/react_agent.py` |
| §3.2 ORA loop (novel contribution)                 | `grounded_vla/agents/ora_agent.py`, `agents/prompts.py`   |
| §3.3 Mind2Web loader                               | `grounded_vla/data/mind2web.py`                           |
| §3.3 ScienceQA loader                              | `grounded_vla/data/scienceqa.py`                          |
| §3.3 Synthetic dataset + two-person review         | `grounded_vla/synthetic/{builder,review}.py`              |
| §3.4 Baselines + Method + Ablation                 | `configs/*.yaml`, `scripts/run_full_eval.py`              |
| §3.4 Metrics (completion, step efficiency, errors) | `grounded_vla/eval/metrics.py`, `error_analysis.py`       |
| H3 stretch: LoRA                                   | `grounded_vla/lora.py`                                    |

## Designing experiments

The three hypotheses map to three pairwise comparisons:

- **H1 (Visual Grounding):** compare `react` vs `single_shot` on Mind2Web
  and the synthetic set. Same reasoning budget (one call, no loop), only
  difference is the presence of vision.
- **H2 (Structured Reasoning):** compare `single_shot` vs `ora` on
  Mind2Web and ScienceQA. Same backbone and vision input; only difference
  is the agentic loop with per-step re-encoding.
- **H3 (PEFT Adaptation — stretch):** compare `ora` vs `ora + LoRA
  adapter` on the synthetic held-out split and Mind2Web. Requires
  `grounded_vla.lora.train_lora` and GPU time.

## Tests

```bash
pip install -e .[dev]
pytest
```

Tests cover: action parser (JSON + NL fallback), the ORA loop (re-encoding
changes outputs), the ReAct loop (parsing failures surface correctly),
metrics (answer normalization), and an end-to-end smoke run.
