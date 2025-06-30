# Context Engineering
Bringing You the Latest Research on Context With First Principles & Visuals — June 2025 from ICML, IBM, NeurIPS, OHBM, and more 

<div align="center">

## [ICML June 2025](https://openreview.net/forum?id=y1SnRPDWx4) | [IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115) | [Indiana June 2025](https://arxiv.org/pdf/2506.10077)

## [Devin DeepWiki Docs](https://deepwiki.com/davidkimai/Context-Engineering)
## [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/davidkimai/Context-Engineering)
</div>

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

> **"Meaning is not an intrinsic, static property of a semantic expression, but rather an emergent phenomenon actualized through the dynamic interaction between the expression and an interpretive agent situated within a specific context."
— [Agostino et al., — June 2025, Indiana University](https://arxiv.org/pdf/2506.10077)**

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

## **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)**

![image](https://github.com/user-attachments/assets/76c6e6cb-b65d-4af7-95a5-6d52aee7efc0)

> **TL;DR: A three-stage architecture is identified that supports abstract reasoning in LLMs via a set of emergent symbol-processing mechanisms.**
>
>
These include symbolic induction heads, symbolic abstraction heads, and retrieval heads.


![image](https://github.com/user-attachments/assets/2428544e-332a-4e32-9070-9f9d8716d491)


>
> **Concept: Leverage symbols/syntax as structural attractors or templates for user guidance of context/reasoning/memory/persistence**

## Under Construction

```python
Context-Engineering/
├── LICENSE                          # MIT license
├── README.md                        # Quick-start overview
├── structure.md                     # Original structural map
├── STRUCTURE_v2.md                  # Enhanced structural map with field theory
├── context.json                     # Original schema configuration
├── context_v2.json                  # Extended schema with field protocols
├── context_v3.json                  # Neural field extensions
├── context_v3.5.json                # Symbolic mechanism integration
├── CITATIONS.md                     # Research references and bridges
│
├── 00_foundations/                  # First-principles theory
│   ├── 01_atoms_prompting.md        # Atomic instruction units
│   ├── 02_molecules_context.md      # Few-shot examples/context
│   ├── 03_cells_memory.md           # Stateful conversation layers
│   ├── 04_organs_applications.md    # Multi-step control flows
│   ├── 05_cognitive_tools.md        # Mental model extensions
│   ├── 06_advanced_applications.md  # Real-world implementations
│   ├── 07_prompt_programming.md     # Code-like reasoning patterns
│   ├── 08_neural_fields_foundations.md # Context as continuous fields
│   ├── 09_persistence_and_resonance.md # Field dynamics and attractors
│   ├── 10_field_orchestration.md    # Coordinating multiple fields
│   ├── 11_emergence_and_attractor_dynamics.md # Emergent properties
│   └── 12_symbolic_mechanisms.md    # Symbolic reasoning in LLMs
│
├── 10_guides_zero_to_hero/          # Hands-on tutorials
│   ├── 01_min_prompt.ipynb          # Minimal prompt experiments
│   ├── 02_expand_context.ipynb      # Context expansion techniques
│   ├── 03_control_loops.ipynb       # Flow control mechanisms
│   ├── 04_rag_recipes.ipynb         # Retrieval-augmented patterns
│   ├── 05_protocol_bootstrap.ipynb  # Field protocol bootstrap
│   ├── 06_protocol_token_budget.ipynb # Protocol efficiency
│   ├── 07_streaming_context.ipynb   # Real-time context
│   ├── 08_emergence_detection.ipynb # Detecting emergence
│   ├── 09_residue_tracking.ipynb    # Tracking symbolic residue
│   └── 10_attractor_formation.ipynb # Creating field attractors
│
├── 20_templates/                    # Reusable components
│   ├── minimal_context.yaml         # Base context structure
│   ├── control_loop.py              # Orchestration template
│   ├── scoring_functions.py         # Evaluation metrics
│   ├── prompt_program_template.py   # Program structure template
│   ├── schema_template.yaml         # Schema definition template
│   ├── recursive_framework.py       # Recursive context template
│   ├── field_protocol_shells.py     # Field protocol templates
│   ├── symbolic_residue_tracker.py  # Residue tracking tools
│   ├── context_audit.py             # Context analysis tool
│   ├── shell_runner.py              # Protocol shell runner
│   ├── resonance_measurement.py     # Field resonance metrics
│   ├── attractor_detection.py       # Attractor analysis tools
│   ├── boundary_dynamics.py         # Boundary operation tools
│   └── emergence_metrics.py         # Emergence measurement
│
├── 30_examples/                     # Practical implementations
│   ├── 00_toy_chatbot/              # Simple conversation agent
│   ├── 01_data_annotator/           # Data labeling system
│   ├── 02_multi_agent_orchestrator/ # Agent collaboration system
│   ├── 03_vscode_helper/            # IDE integration 
│   ├── 04_rag_minimal/              # Minimal RAG implementation
│   ├── 05_streaming_window/         # Real-time context demo
│   ├── 06_residue_scanner/          # Symbolic residue demo
│   ├── 07_attractor_visualizer/     # Field visualization
│   ├── 08_field_protocol_demo/      # Protocol demonstration
│   └── 09_emergence_lab/            # Emergence experimentation
│
├── 40_reference/                    # Deep-dive documentation
│   ├── token_budgeting.md           # Token optimization strategies
│   ├── retrieval_indexing.md        # Retrieval system design
│   ├── eval_checklist.md            # PR evaluation criteria
│   ├── cognitive_patterns.md        # Reasoning pattern catalog
│   ├── schema_cookbook.md           # Schema pattern collection
│   ├── patterns.md                  # Context pattern library
│   ├── field_mapping.md             # Field theory fundamentals
│   ├── symbolic_residue_types.md    # Residue classification
│   ├── attractor_dynamics.md        # Attractor theory and practice
│   ├── emergence_signatures.md      # Detecting emergence
│   └── boundary_operations.md       # Boundary management guide
│
├── 50_contrib/                      # Community contributions
│   └── README.md                    # Contribution guidelines
│
├── 60_protocols/                    # Protocol shells and frameworks
│   ├── README.md                    # Protocol overview
│   ├── shells/                      # Protocol shell definitions
│   │   ├── attractor.co.emerge.shell      # Attractor co-emergence
│   │   ├── recursive.emergence.shell      # Recursive field emergence
│   │   ├── recursive.memory.attractor.shell # Memory persistence
│   │   ├── field.resonance.scaffold.shell  # Field resonance
│   │   ├── field.self_repair.shell        # Self-repair mechanisms
│   │   └── context.memory.persistence.attractor.shell # Context persistence
│   ├── digests/                     # Simplified protocol documentation
│   └── schemas/                     # Protocol schemas
│       ├── fractalRepoContext.v3.5.json    # Repository context
│       ├── fractalConsciousnessField.v1.json # Field schema
│       ├── protocolShell.v1.json           # Shell schema
│       ├── symbolicResidue.v1.json         # Residue schema
│       └── attractorDynamics.v1.json       # Attractor schema
│
├── 70_agents/                       # Agent demonstrations
│   ├── README.md                    # Agent overview
│   ├── 01_residue_scanner/          # Symbolic residue detection
│   ├── 02_self_repair_loop/         # Self-repair protocol
│   ├── 03_attractor_modulator/      # Attractor dynamics
│   ├── 04_boundary_adapter/         # Dynamic boundary tuning
│   └── 05_field_resonance_tuner/    # Field resonance optimization
│
├── 80_field_integration/            # Complete field projects
│   ├── README.md                    # Integration overview
│   ├── 00_protocol_ide_helper/      # Protocol development tools
│   ├── 01_context_engineering_assistant/ # Field-based assistant
│   ├── 02_recursive_reasoning_system/    # Recursive reasoning
│   ├── 03_emergent_field_laboratory/     # Field experimentation
│   └── 04_symbolic_reasoning_engine/     # Symbolic mechanisms
│
├── cognitive-tools/                 # Advanced cognitive framework
│   ├── README.md                    # Overview and quick-start guide
│   ├── cognitive-templates/         # Templates for reasoning
│   │   ├── understanding.md         # Comprehension operations
│   │   ├── reasoning.md             # Analytical operations
│   │   ├── verification.md          # Checking and validation
│   │   ├── composition.md           # Combining multiple tools
│   │   └── emergence.md             # Emergent reasoning patterns
│   │
│   ├── cognitive-programs/          # Structured prompt programs
│   │   ├── basic-programs.md        # Fundamental program structures
│   │   ├── advanced-programs.md     # Complex program architectures
│   │   ├── program-library.py       # Python implementations
│   │   ├── program-examples.ipynb   # Interactive examples
│   │   └── emergence-programs.md    # Emergent program patterns
│   │
│   ├── cognitive-schemas/           # Knowledge representations
│   │   ├── user-schemas.md          # User information schemas
│   │   ├── domain-schemas.md        # Domain knowledge schemas
│   │   ├── task-schemas.md          # Reasoning task schemas
│   │   ├── schema-library.yaml      # Reusable schema library
│   │   └── field-schemas.md         # Field representation schemas
│   │
│   ├── cognitive-architectures/     # Complete reasoning systems
│   │   ├── solver-architecture.md   # Problem-solving systems
│   │   ├── tutor-architecture.md    # Educational systems
│   │   ├── research-architecture.md # Information synthesis
│   │   ├── architecture-examples.py # Implementation examples
│   │   └── field-architecture.md    # Field-based architectures
│   │
│   └── integration/                 # Integration patterns
│       ├── with-rag.md              # Integration with retrieval
│       ├── with-memory.md           # Integration with memory
│       ├── with-agents.md           # Integration with agents
│       ├── evaluation-metrics.md    # Effectiveness measurement
│       └── with-fields.md           # Integration with field protocols
│
└── .github/                         # GitHub configuration
    ├── CONTRIBUTING.md              # Contribution guidelines
    ├── workflows/ci.yml             # CI pipeline configuration
    ├── workflows/eval.yml           # Evaluation automation
    └── workflows/protocol_tests.yml # Protocol testing
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
| **Neural Field Theory** | Context as a Neural Field | Modeling context as a dynamic neural field allows for iterative context updating |
| **Symbolic Mechnanisms** | Symbolic architectures enable higher order reasoning | Smarter systems = less work |
| **Quantum Semantics** |  Meaning as observer-dependent  | Design context systems leveraging superpositional techniques |



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
