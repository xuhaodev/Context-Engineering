# Context Engineering
## [Eliciting Reasoning in Language Models with Cognitive Tools — IBM June 2025](https://www.arxiv.org/pdf/2506.12115)
> **"Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — [**Andrej Karpathy**](https://x.com/karpathy/status/1937902205765607626)**

A practical, first-principles handbook for moving beyond prompt engineering to the wider discipline of context design, orchestration, and optimization.

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```


## Why This Repository Exists

> **We posit that meaning is instead actualized through an observer-dependent interpretive act - [Indiana University June 2025](https://arxiv.org/pdf/2506.10077)**

Prompt engineering gets all the attention, but we can now get excited for what comes next. Once you've mastered prompts, the real power comes from engineering the **entire context window** that surrounds those prompts. Guiding thought, if you will. 

This repository provides a progressive, first-principles approach to context engineering, built around a biological metaphor:

```
atoms → molecules → cells → organs → neurobiological systems → neural field theory
  │        │         │        │             │                         │        
single    few-     memory   multi-    cognitive tools +   context = neural fields +
prompt    shot     state    agent     prompt programs     persistence & resonance
```
> "Abstraction is the cost of generalization"— [**Grant Sanderson (3Blue1Brown)**](https://www.3blue1brown.com/)



## Symbols? 

> **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML 2025](https://openreview.net/forum?id=y1SnRPDWx4)**
>
> **TL;DR: A three-stage architecture is identified that supports abstract reasoning in LLMs via a set of emergent symbol-processing mechanisms**
> 
>
> **Concept: Leverage symbols/syntax as structural attractors for context/reasoning/memory/persistence**

## Under Construction

```
Context-Engineering/
├── LICENSE                          # MIT license
├── README.md                        # Quick-start overview
├── structure.md                     # Core structural map
├── context.json                     # Schema configuration
│
├── 00_foundations/                  # First-principles theory
│   ├── 01_atoms_prompting.md        # Atomic instruction units
│   ├── 02_molecules_context.md      # Few-shot examples/context
│   ├── 03_cells_memory.md           # Stateful conversation layers
│   ├── 04_organs_applications.md    # Multi-step control flows
│   ├── 05_cognitive_tools.md        # Mental model extensions
│   ├── 06_advanced_applications.md  # Real-world implementations
│   └── 07_prompt_programming.md     # Code-like reasoning patterns
│
├── 10_guides_zero_to_hero/           # Hands-on tutorials
│   ├── 01_min_prompt.ipynb          # Minimal prompt experiments
│   ├── 02_expand_context.ipynb      # Context expansion techniques
│   ├── 03_control_loops.ipynb       # Flow control mechanisms
│   ├── 04_rag_recipes.ipynb         # Retrieval-augmented patterns
│   ├── 05_prompt_programs.ipynb     # Structured reasoning programs
│   ├── 06_schema_design.ipynb       # Schema creation patterns
│   └── 07_recursive_patterns.ipynb  # Self-referential contexts
│
├── 20_templates/                    # Reusable components
│   ├── minimal_context.yaml         # Base context structure
│   ├── control_loop.py              # Orchestration template
│   ├── scoring_functions.py         # Evaluation metrics
│   ├── prompt_program_template.py   # Program structure template
│   ├── schema_template.yaml         # Schema definition template
│   └── recursive_context.py       # Recursive context template
│
├── 30_examples/                     # Practical implementations
│   ├── 00_toy_chatbot/              # Simple conversation agent
│   ├── 01_data_annotator/           # Data labeling system
│   ├── 02_multi_agent_orchestrator/ # Agent collaboration system
│   ├── 03_cognitive_assistant/      # Advanced reasoning assistant
│   └── 04_rag_minimal/              # Minimal RAG implementation
│
├── 40_reference/                    # Deep-dive documentation
│   ├── token_budgeting.md           # Token optimization strategies
│   ├── retrieval_indexing.md        # Retrieval system design
│   ├── eval_checklist.md            # PR evaluation criteria
│   ├── cognitive_patterns.md        # Reasoning pattern catalog
│   └── schema_cookbook.md           # Schema pattern collection
│
├── 50_contrib/                      # Community contributions
│   └── README.md                    # Contribution guidelines
│
├── cognitive-tools/                 # Advanced cognitive framework
│   ├── README.md                    # Overview
│   ├── cognitive-templates/         # Templates for reasoning
│   ├── cognitive-programs/          # Program implementations
│   ├── cognitive-schemas/           # Schema definitions
│   ├── cognitive-architectures/     # Full system designs
│   └── integration/                 # Integration patterns
│
└── .github/                         # GitHub configuration
    ├── CONTRIBUTING.md              # Contribution guidelines
    └── workflows/ci.yml             # CI pipeline configuration
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
| **Cognitive Tools & Prompt Programming** | Learm to build custom tools and templates | Prompt programming enables new layers for context engineering |

## Karpathy + 3Blue1Brown Inspired Style

> For learners of all experience levels

1. **First principles** – start with the fundamental context
2. **Iterative add-on** – add only what the model demonstrably lacks
3. **Measure everything** – token cost, latency, quality score
4. **Delete ruthlessly** – pruning beats padding
5. **Code > slides** – every concept has a runnable cell
6. **Visualize everything** — every concept is visualized with ASCII and symbolic diagrams

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
  url = {https://github.com/davidkimai/context-engineering}
}
```

## Acknowledgements
> I've been looking forward to this being conceptualized and formalized as there wasn't a prior established field. Prompt engineering receives quite the stigma and doesn't quite cover what most researchers and I do.

- [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo 
- All contributors and the open source community
