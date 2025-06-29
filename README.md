# Context-Engineering

> "Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)

A practical, first-principles handbook for moving beyond prompt engineering to the wider discipline of context design, orchestration, and optimization.

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```
## Under Construction

```
Context-Engineering/
├── LICENSE
├── README.md                  # 2-minute overview + quick-start links
├── structure.md               # *← copy the markdown content provided below*
│
├── 00_foundations/            # first-principles theory (atoms → molecules → cells)
│   ├── 01_atoms_prompting.md
│   ├── 02_molecules_context.md
│   ├── 03_cells_memory.md
│   └── 04_organs_applications.md
│
├── 10_guides_zero_to_hero/     # hands-on walkthroughs
│   ├── 01_min_prompt.ipynb
│   ├── 02_expand_context.ipynb
│   ├── 03_control_loops.ipynb
│   └── 04_rag_recipes.ipynb
│
├── 20_templates/              # reusable boilerplate
│   ├── minimal_context.yaml
│   ├── control_loop.py
│   └── scoring_functions.py
│
├── 30_examples/               # real projects, progressively complex
│   ├── 00_toy_chatbot/
│   ├── 01_data_annotator/
│   └── 02_multi_agent_orchestrator/
│
├── 40_reference/              # “Karpathy-style” deep dives
│   ├── token_budgeting.md
│   ├── retrieval_indexing.md
│   └── eval_checklist.md
│
├── 50_contrib/                # community PRs live here
│   └── README.md
│
└── .github/
    ├── CONTRIBUTING.md
    └── workflows/ci.yml

```
## Why This Repository Exists

Prompt engineering gets all the attention, but imagine what comes next. Once you've mastered prompts, the real power comes from engineering the **entire context window** that surrounds those prompts.

This repository provides a progressive, first-principles approach to context engineering, built around a biological metaphor:

```
atoms → molecules → cells → organs → systems
  │        │         │        │         │
single    few-     memory   multi-    full
prompt    shot     state    agent     apps
```

## Quick Start

1. **Read `00_foundations/01_atoms_prompting.md`** (5 min)  
   Understand why prompts alone often underperform

2. **Run `10_guides_zero_to_one/01_min_prompt.py (Jupyter Notebook style)`**  
   Experiment with a minimal working example

3. **Explore `20_templates/minimal_context.yaml`**  
   Copy/paste a template into your own project  

4. **Study `30_examples/00_toy_chatbot/`**  
   See a complete implementation with context management

## Learning Path

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│ 00_foundations/ │     │ 10_guides_zero_  │     │ 20_templates/  │
│                 │────▶│ to_one/          │────▶│                │
│ Theory & core   │     │ Hands-on         │     │ Copy-paste     │
│ concepts        │     │ walkthroughs     │     │ snippets       │
└─────────────────┘     └──────────────────┘     └────────────────┘
         │                                                │
         │                                                │
         ▼                                                ▼
┌─────────────────┐                             ┌────────────────┐
│ 40_reference/   │◀───────────────────────────▶│ 30_examples/   │
│                 │                             │                │
│ Deep dives &    │                             │ Real projects, │
│ eval cookbook   │                             │ progressively  │
└─────────────────┘                             │ complex        │
         ▲                                      └────────────────┘
         │                                                ▲
         │                                                │
         └────────────────────┐               ┌───────────┘
                              ▼               ▼
                         ┌─────────────────────┐
                         │ 50_contrib/         │
                         │                     │
                         │ Community           │
                         │ contributions       │
                         └─────────────────────┘
```

## What You'll Learn

| Concept | What It Is | Why It Matters |
|---------|------------|----------------|
| **Token Budget** | Optimizing every token in your context | More tokens = more $$ and slower responses |
| **Few-Shot Learning** | Teaching by showing examples | Often works better than explanation alone |
| **Memory Systems** | Persisting information across turns | Enables stateful, coherent interactions |
| **Retrieval Augmentation** | Finding & injecting relevant documents | Grounds responses in facts, reduces hallucination |
| **Control Flow** | Breaking complex tasks into steps | Solve harder problems with simpler prompts |
| **Context Pruning** | Removing irrelevant information | Keep only what's necessary for performance |
| **Metrics & Evaluation** | Measuring context effectiveness | Iterative optimization of token use vs. quality |

## Karpathy-Style Guidelines

1. **Minimal first pass** – start with the smallest viable context
2. **Iterative add-on** – add only what the model demonstrably lacks
3. **Measure everything** – token cost, latency, quality score
4. **Delete ruthlessly** – pruning beats padding
5. **Code > slides** – every concept has a runnable cell

## Contributing

We welcome contributions! Check out [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## License

[MIT License](LICENSE)

## Citation

```bibtex
@misc{context-engineering,
  author = {Context Engineering Contributors},
  title = {Context Engineering: Beyond Prompt Engineering},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-username/context-engineering}
}
```

## Acknowledgements

- [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring the approach
- All contributors and the open source community
