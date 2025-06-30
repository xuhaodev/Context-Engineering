# Pareto-lang: A Declarative Language for Context Operations

> *"Give me a lever long enough and a fulcrum on which to place it, and I shall move the world."*
>
>
> **— Archimedes**

## 1. Introduction: The Power of Operational Grammar

In our journey through context engineering, we've explored protocol shells as templates for organizing AI communication. Now, we delve into Pareto-lang – a powerful, declarative grammar designed specifically for performing operations on context.

Pareto-lang is named after Vilfredo Pareto, the economist who identified the 80/20 principle – the idea that roughly 80% of effects come from 20% of causes. In the realm of context engineering, Pareto-lang embodies this principle by providing a minimal but powerful syntax that enables sophisticated context operations with remarkable efficiency.

**Socratic Question**: Think about command languages you've encountered – from command-line interfaces to search query syntax. What makes some more intuitive and powerful than others? How might a specialized grammar for context operations transform how you interact with AI?

```
┌─────────────────────────────────────────────────────────┐
│                  PARETO-LANG ESSENCE                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Protocol Shells            Pareto-lang                 │
│  ───────────────           ───────────                  │
│  Define structure          Define operations            │
│  ↓                         ↓                            │
│                                                         │
│  /protocol.name{           /operation.modifier{         │
│    intent="...",             parameter="value",         │
│    input={...},              target="element"           │
│    process=[...],          }                            │
│    output={...}                                         │
│  }                                                      │
│                                                         │
│  Containers for            Actions that transform       │
│  organizing communication  context elements             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 2. Pareto-lang: Core Syntax and Structure

At its core, Pareto-lang offers a simple, consistent syntax for describing operations:

```
/operation.modifier{parameters}
```

This deceptively simple format enables a wide range of powerful context operations.

### 2.1. Anatomy of a Pareto-lang Operation

Let's break down the components:

```
┌─────────────────────────────────────────────────────────┐
│                 PARETO-LANG ANATOMY                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  /compress.summary{target="history", method="key_points"}
│   │        │       │        │        │        │
│   │        │       │        │        │        └── Value
│   │        │       │        │        │
│   │        │       │        │        └── Parameter name
│   │        │       │        │
│   │        │       │        └── Parameter opening
│   │        │       │
│   │        │       └── Parameters block opening
│   │        │
│   │        └── Operation subtype or variant
│   │
│   └── Core operation
│
└─────────────────────────────────────────────────────────┘
```

Each element serves a specific purpose:

1. **Core Operation (`/compress`)**: The primary action to be performed.
2. **Operation Modifier (`.summary`)**: A qualifier that specifies the variant or method of the operation.
3. **Parameters Block (`{...}`)**: Contains the configuration details for the operation.
4. **Parameter Names and Values**: Key-value pairs that precisely control how the operation executes.

### 2.2. Basic Syntax Rules

Pareto-lang follows a few simple but strict rules:

1. **Forward Slash Prefix**: All operations begin with a forward slash (`/`).
2. **Dot Notation**: The core operation and modifier are separated by a dot (`.`).
3. **Curly Braces**: Parameters are enclosed in curly braces (`{` and `}`).
4. **Key-Value Pairs**: Parameters are specified as `key="value"` or `key=value`.
5. **Commas**: Multiple parameters are separated by commas.
6. **Quotes**: String values are enclosed in quotes, while numbers and booleans are not.

### 2.3. Nesting and Composition

Pareto-lang operations can be nested within each other for complex operations:

```
/operation1.modifier1{
    param1="value1",
    nested=/operation2.modifier2{
        param2="value2"
    }
}
```

They can also be composed into sequences within protocol shells:

```
process=[
    /operation1.modifier1{params...},
    /operation2.modifier2{params...},
    /operation3.modifier3{params...}
]
```

**Reflective Exercise**: Look at the structure of Pareto-lang. How does its simplicity and consistency make it both accessible to beginners and powerful for advanced users?

## 3. Core Operation Categories

Pareto-lang operations fall into several functional categories, each addressing different aspects of context management:

```
┌─────────────────────────────────────────────────────────┐
│                 OPERATION CATEGORIES                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Information       ┌──────────────────────┐             │
│  Management        │ /extract, /filter,   │             │
│                    │ /prioritize, /group  │             │
│                    └──────────────────────┘             │
│                                                         │
│  Content           ┌──────────────────────┐             │
│  Transformation    │ /compress, /expand,  │             │
│  and Optimization  │ /restructure, /format│             │
│                    └──────────────────────┘             │
│                                                         │
│  Analysis and      ┌──────────────────────┐             │
│  Insight Generation│ /analyze, /evaluate, │             │
│                    │ /compare, /synthesize│             │
│                    └──────────────────────┘             │
│                                                         │
│  Field             ┌──────────────────────┐             │
│  Operations        │ /attractor, /boundary,│             │
│                    │ /resonance, /residue │             │
│                    └──────────────────────┘             │
│                                                         │
│  Memory and        ┌──────────────────────┐             │
│  State Management  │ /remember, /forget,  │             │
│                    │ /update, /retrieve   │             │
│                    └──────────────────────┘             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

Let's explore each category in detail.

## 4. Information Management Operations

Information management operations help you control what information is included, excluded, or emphasized in your context.

### 4.1. Extract Operations

Extract operations pull specific information from larger content:

```
/extract.key_points{
    from="document",
    focus=["main arguments", "supporting evidence", "conclusions"],
    method="semantic_clustering",
    max_points=7
}
```

Common variants:
- `/extract.key_points`: Extract main points or ideas
- `/extract.entities`: Extract named entities (people, places, organizations)
- `/extract.relationships`: Extract relationships between elements
- `/extract.metrics`: Extract quantitative measures or statistics

### 4.2. Filter Operations

Filter operations remove or include information based on criteria:

```
/filter.relevance{
    threshold=0.7,
    criteria="relevance_to_query",
    preserve="high_value_information",
    exclude="tangential_details"
}
```

Common variants:
- `/filter.relevance`: Filter based on relevance to a topic or query
- `/filter.recency`: Filter based on how recent information is
- `/filter.importance`: Filter based on importance or significance
- `/filter.uniqueness`: Filter to remove redundancy

### 4.3. Prioritize Operations

Prioritize operations rank information by importance:

```
/prioritize.importance{
    criteria=["relevance", "impact", "urgency"],
    weighting=[0.5, 0.3, 0.2],
    top_n=5,
    include_scores=true
}
```

Common variants:
- `/prioritize.importance`: Rank by overall importance
- `/prioritize.relevance`: Rank by relevance to current topic
- `/prioritize.impact`: Rank by potential impact or significance
- `/prioritize.urgency`: Rank by time sensitivity

### 4.4. Group Operations

Group operations organize information into logical clusters:

```
/group.category{
    elements="document_sections",
    by="topic",
    max_groups=5,
    allow_overlap=false
}
```

Common variants:
- `/group.category`: Group by categorical attributes
- `/group.similarity`: Group by semantic similarity
- `/group.hierarchy`: Group into hierarchical structure
- `/group.chronology`: Group by temporal sequence

**Socratic Question**: Which information management operations would be most valuable for your typical AI interactions? How might explicit filtering or prioritization change the quality of responses you receive?

## 5. Content Transformation and Optimization Operations

These operations modify content to improve clarity, efficiency, or effectiveness.

### 5.1. Compress Operations

Compress operations reduce content size while preserving key information:

```
/compress.summary{
    target="conversation_history",
    ratio=0.3,
    method="extractive",
    preserve=["decisions", "key_facts", "action_items"]
}
```

Common variants:
- `/compress.summary`: Create a condensed summary
- `/compress.key_value`: Extract and store as key-value pairs
- `/compress.outline`: Create a hierarchical outline
- `/compress.abstractive`: Generate a new, condensed version

### 5.2. Expand Operations

Expand operations elaborate on or develop content:

```
/expand.detail{
    topic="technical_concept",
    aspects=["definition", "examples", "applications", "limitations"],
    depth="comprehensive",
    style="educational"
}
```

Common variants:
- `/expand.detail`: Add more detailed information
- `/expand.example`: Add illustrative examples
- `/expand.clarification`: Add explanatory information
- `/expand.implication`: Explore consequences or implications

### 5.3. Restructure Operations

Restructure operations reorganize content for clarity or effectiveness:

```
/restructure.format{
    content="technical_explanation",
    structure="step_by_step",
    components=["concept", "example", "application", "caution"],
    flow="logical_progression"
}
```

Common variants:
- `/restructure.format`: Change the overall format
- `/restructure.sequence`: Change the order of elements
- `/restructure.hierarchy`: Reorganize hierarchical relationships
- `/restructure.grouping`: Reorganize how elements are grouped

### 5.4. Format Operations

Format operations change how content is presented:

```
/format.style{
    target="document",
    style="academic",
    elements=["headings", "citations", "terminology"],
    consistency=true
}
```

Common variants:
- `/format.style`: Change the writing or presentation style
- `/format.layout`: Change the visual organization
- `/format.highlight`: Emphasize key elements
- `/format.simplify`: Make content more accessible

**Reflective Exercise**: Consider a recent complex document or conversation. Which transformation operations would help make it more clear, concise, or effective? How would you specify the parameters to get exactly the transformation you need?

## 6. Analysis and Insight Generation Operations

These operations help extract meaning, patterns, and insights from content.

### 6.1. Analyze Operations

Analyze operations examine content to understand its structure, components, or meaning:

```
/analyze.structure{
    content="academic_paper",
    identify=["sections", "arguments", "evidence", "methodology"],
    depth="comprehensive",
    approach="systematic"
}
```

Common variants:
- `/analyze.structure`: Examine organizational structure
- `/analyze.argument`: Examine logical structure and validity
- `/analyze.sentiment`: Examine emotional tone or attitude
- `/analyze.trend`: Examine patterns over time
- `/analyze.relationship`: Examine connections between elements

### 6.2. Evaluate Operations

Evaluate operations assess quality, validity, or effectiveness:

```
/evaluate.evidence{
    claims=["claim1", "claim2", "claim3"],
    criteria=["relevance", "credibility", "sufficiency"],
    scale="1-5",
    include_justification=true
}
```

Common variants:
- `/evaluate.evidence`: Assess supporting evidence
- `/evaluate.argument`: Assess logical strength
- `/evaluate.source`: Assess credibility or reliability
- `/evaluate.impact`: Assess potential consequences
- `/evaluate.performance`: Assess effectiveness or efficiency

### 6.3. Compare Operations

Compare operations identify similarities, differences, or relationships:

```
/compare.concepts{
    items=["concept1", "concept2", "concept3"],
    dimensions=["definition", "examples", "applications", "limitations"],
    method="side_by_side",
    highlight_differences=true
}
```

Common variants:
- `/compare.concepts`: Compare ideas or theories
- `/compare.options`: Compare alternatives or choices
- `/compare.versions`: Compare different versions or iterations
- `/compare.perspectives`: Compare different viewpoints

### 6.4. Synthesize Operations

Synthesize operations combine information to generate new insights:

```
/synthesize.insights{
    sources=["research_papers", "expert_opinions", "market_data"],
    framework="integrated_analysis",
    focus="emerging_patterns",
    generate_implications=true
}
```

Common variants:
- `/synthesize.insights`: Generate new understanding
- `/synthesize.framework`: Create organizing structure
- `/synthesize.theory`: Develop explanatory model
- `/synthesize.recommendation`: Develop action-oriented guidance

**Socratic Question**: How might explicit analysis operations help you gain deeper insights from complex information? Which synthesis operations would be most valuable for your decision-making processes?

## 7. Field Operations

Field operations apply concepts from field theory to manage context as a continuous semantic landscape.

### 7.1. Attractor Operations

Attractor operations manage semantic focal points in the field:

```
/attractor.identify{
    field="conversation_context",
    method="semantic_density_mapping",
    threshold=0.7,
    max_attractors=5
}
```

Common variants:
- `/attractor.identify`: Detect semantic attractors
- `/attractor.strengthen`: Increase attractor influence
- `/attractor.weaken`: Decrease attractor influence
- `/attractor.create`: Establish new semantic attractors
- `/attractor.merge`: Combine related attractors

### 7.2. Boundary Operations

Boundary operations control information flow and field delineation:

```
/boundary.establish{
    around="topic_cluster",
    permeability=0.6,
    criteria="semantic_relevance",
    gradient=true
}
```

Common variants:
- `/boundary.establish`: Create information boundaries
- `/boundary.adjust`: Modify existing boundaries
- `/boundary.dissolve`: Remove boundaries
- `/boundary.filter`: Control what crosses boundaries

### 7.3. Resonance Operations

Resonance operations manage how elements interact and reinforce each other:

```
/resonance.amplify{
    between=["concept1", "concept2"],
    method="explicit_connection",
    strength=0.8,
    bi_directional=true
}
```

Common variants:
- `/resonance.detect`: Identify pattern relationships
- `/resonance.amplify`: Strengthen connections
- `/resonance.dampen`: Weaken connections
- `/resonance.harmonize`: Create coherent pattern relationships

### 7.4. Residue Operations

Residue operations handle persistent fragments of meaning:

```
/residue.track{
    types=["key_definitions", "recurring_themes", "emotional_tones"],
    persistence="across_context_windows",
    integration=true
}
```

Common variants:
- `/residue.track`: Monitor symbolic fragments
- `/residue.preserve`: Maintain important residue
- `/residue.integrate`: Incorporate residue into field
- `/residue.clear`: Remove unwanted residue

```
┌─────────────────────────────────────────────────────────┐
│                FIELD OPERATIONS MAP                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│         Attractor Basin                 Boundary        │
│             ╱─╲                          ┌┈┈┈┐          │
│            /   \                         ┊   ┊          │
│           /     \         Resonance      ┊   ┊          │
│     ┈┈┈┈┈┘       └┈┈┈┈    ↔↔↔↔↔↔↔↔       ┊   ┊          │
│                                          ┊   ┊          │
│     Attractor    Attractor               ┊   ┊          │
│       ╱─╲          ╱─╲                   ┊   ┊          │
│      /   \        /   \                  ┊   ┊          │
│     /     \      /     \                 ┊   ┊          │
│ ┈┈┈┘       └┈┈┈┈┘       └┈┈┈┈            └┈┈┈┘          │
│                                                         │
│                    Residue                              │
│                      •                                  │
│                    •   •                                │
│                  •       •                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Reflective Exercise**: Consider your understanding of field theory concepts. How might these operations help you manage complex, evolving contexts? Which field operations would be most useful for maintaining coherence in extended conversations?

## 8. Memory and State Management Operations

These operations help manage information persistence across interactions.

### 8.1. Remember Operations

Remember operations store information for future reference:

```
/remember.key_value{
    key="user_preference",
    value="dark_mode",
    persistence="session",
    priority="high"
}
```

Common variants:
- `/remember.key_value`: Store as key-value pairs
- `/remember.context`: Store contextual information
- `/remember.decision`: Store choices or decisions
- `/remember.insight`: Store important realizations

### 8.2. Forget Operations

Forget operations remove information from active memory:

```
/forget.outdated{
    older_than="30_days",
    categories=["temporary_notes", "resolved_issues"],
    confirmation=true
}
```

Common variants:
- `/forget.outdated`: Remove old information
- `/forget.irrelevant`: Remove information no longer needed
- `/forget.superseded`: Remove information that has been replaced
- `/forget.sensitive`: Remove private or sensitive information

### 8.3. Update Operations

Update operations modify stored information:

```
/update.information{
    key="project_status",
    old_value="in_progress",
    new_value="completed",
    timestamp=true
}
```

Common variants:
- `/update.information`: Change stored information
- `/update.priority`: Change importance level
- `/update.status`: Change state or status
- `/update.relationship`: Change how information relates to other elements

### 8.4. Retrieve Operations

Retrieve operations access stored information:

```
/retrieve.memory{
    key="previous_discussion",
    related_to="current_topic",
    max_items=3,
    format="summary"
}
```

Common variants:
- `/retrieve.memory`: Access stored information
- `/retrieve.history`: Access conversation history
- `/retrieve.decision`: Access previous choices
- `/retrieve.preference`: Access user preferences

**Socratic Question**: How would explicit memory operations change your long-running interactions with AI? What types of information would be most valuable to explicitly remember, update, or forget?

## 9. Advanced Pareto-lang Features

Beyond basic operations, Pareto-lang includes several advanced features for complex context management.

### 9.1. Conditional Operations

Conditional operations execute based on specific conditions:

```
/if.condition{
    test="token_count > 4000",
    then=/compress.summary{target="history", ratio=0.5},
    else=/maintain.current{target="history"}
}
```

Structure:
- `test`: The condition to evaluate
- `then`: Operation to execute if condition is true
- `else`: (Optional) Operation to execute if condition is false

### 9.2. Iteration Operations

Iteration operations repeat processing for multiple elements:

```
/for.each{
    items="document_sections",
    do=/analyze.content{
        extract=["key_points", "entities"],
        depth="comprehensive"
    },
    aggregate="combine_results"
}
```

Structure:
- `items`: Collection to iterate over
- `do`: Operation to apply to each item
- `aggregate`: (Optional) How to combine results

### 9.3. Pipeline Operations

Pipeline operations chain multiple operations with data flow:

```
/pipeline.sequence{
    operations=[
        /extract.sections{from="document"},
        /filter.relevance{threshold=0.7},
        /analyze.content{depth="detailed"},
        /synthesize.insights{framework="integrated"}
    ],
    pass_result=true,
    error_handling="continue_with_available"
}
```

Structure:
- `operations`: Sequence of operations to execute
- `pass_result`: Whether to pass results between operations
- `error_handling`: How to handle operation failures

### 9.4. Custom Operation Definition

Define reusable custom operations:

```
/define.operation{
    name="document_analysis",
    parameters=["document", "focus", "depth"],
    implementation=/pipeline.sequence{
        operations=[
            /extract.structure{from=parameter.document},
            /filter.relevance{criteria=parameter.focus},
            /analyze.content{depth=parameter.depth}
        ]
    }
}

// Usage
/document_analysis{
    document="research_paper",
    focus="methodology",
    depth="detailed"
}
```

Structure:
- `name`: Name of the custom operation
- `parameters`: Parameters the operation accepts
- `implementation`: Operation sequence to execute

**Reflective Exercise**: How might these advanced features enable more sophisticated context management? Consider a complex interaction scenario – how would you use conditional operations or pipelines to handle it more effectively?

## 10. Practical Pareto-lang Patterns

Let's explore some practical patterns for common context engineering tasks.

### 10.1. Token Budget Management Pattern

```
/manage.token_budget{
    context_window=8000,
    allocation={
        system=0.15,
        history=0.40,
        current=0.30,
        reserve=0.15
    },
    monitoring=[
        /check.usage{
            component="history",
            if="usage > allocation * 0.9",
            then=/compress.summary{
                target="oldest_messages",
                preserve=["decisions", "key_information"],
                ratio=0.5
            }
        },
        /check.usage{
            component="system",
            if="usage > allocation * 1.1",
            then=/compress.essential{
                target="system_instructions",
                method="priority_based"
            }
        }
    ],
    reporting=true
}
```

### 10.2. Conversation Memory Pattern

```
/manage.conversation_memory{
    strategies=[
        /extract.key_information{
            from="user_messages",
            categories=["preferences", "facts", "decisions"],
            store_as="key_value"
        },
        
        /extract.key_information{
            from="assistant_responses",
            categories=["explanations", "recommendations", "commitments"],
            store_as="key_value"
        },
        
        /track.conversation_state{
            attributes=["topic", "sentiment", "open_questions"],
            update="after_each_exchange"
        },
        
        /manage.history{
            max_messages=10,
            if="exceeded",
            then=/compress.summary{
                target="oldest_messages",
                method="key_points"
            }
        }
    ],
    
    retrieval=[
        /retrieve.relevant{
            to="current_query",
            from="stored_memory",
            max_items=5,
            order="relevance"
        },
        
        /retrieve.state{
            attributes=["current_topic", "open_questions"],
            format="context_prefix"
        }
    ]
}
```

### 10.3. Field-Aware Analysis Pattern

```
/analyze.field_aware{
    content="complex_document",
    
    field_initialization=[
        /field.initialize{
            dimensions=["conceptual", "emotional", "practical"],
            initial_state="neutral"
        },
        
        /attractor.seed{
            from="document_keywords",
            strength=0.7,
            max_attractors=5
        }
    ],
    
    field_analysis=[
        /attractor.evolve{
            iterations=3,
            method="semantic_resonance",
            stabilize=true
        },
        
        /boundary.detect{
            between="concept_clusters",
            threshold=0.6,
            map="gradient_boundaries"
        },
        
        /resonance.measure{
            between="key_concepts",
            strength_threshold=0.7,
            pattern_detection=true
        },
        
        /residue.identify{
            throughout="document",
            types=["persistent_themes", "emotional_undercurrents"],
            significance_threshold=0.6
        }
    ],
    
    insights=[
        /generate.from_attractors{
            focus="dominant_themes",
            depth="significant",
            format="key_points"
        },
        
        /generate.from_boundaries{
            focus="conceptual_divisions",
            interpretation="meaning_of_separations",
            format="analysis"
        },
        
        /generate.from_resonance{
            focus="concept_relationships",
            pattern_significance=true,
            format="network_analysis"
        },
        
        /generate.from_residue{
            focus="underlying_themes",
            implicit_content=true,
            format="deep_insights"
        }
    ]
}
```

### 10.4. Information Extraction and Synthesis Pattern

```
/extract.and.synthesize{
    source="multiple_documents",
    
    extraction=[
        /for.each{
            items="documents",
            do=/extract.key_elements{
                elements=["facts", "arguments", "evidence", "conclusions"],
                method="semantic_parsing",
                confidence_threshold=0.7
            }
        },
        
        /normalize.extracted{
            resolve_conflicts=true,
            standardize_terminology=true,
            remove_duplicates=true
        }
    ],
    
    analysis=[
        /categorize.information{
            scheme="topic_based",
            granularity="medium",
            allow_overlap=true
        },
        
        /identify.patterns{
            types=["trends", "contradictions", "gaps", "consensus"],
            across="all_extracted_information",
            significance_threshold=0.6
        },
        
        /evaluate.quality{
            criteria=["credibility", "relevance", "recency", "comprehensiveness"],
            weight=[0.3, 0.3, 0.2, 0.2]
        }
    ],
    
    synthesis=[
        /integrate.information{
            method="thematic_framework",
            resolution="contradiction_aware",
            level="comprehensive"
        },
        
        /generate.insights{
            based_on=["patterns", "evaluation", "integration"],
            depth="significant",
            perspective="objective"
        },
        
        /structure.output{
            format="progressive_disclosure",
            components=["executive_summary", "key_findings", "detailed_analysis", "implications"],
            navigation="hierarchical"
        }
    ]
}
```

**Socratic Question**: Looking at these patterns, which elements could you adapt for your specific context management needs? How would you modify them to better suit your particular use cases?

## 11. Building Your Own Pareto-lang Operations

Creating effective Pareto-lang operations involves several key steps:

### 11.1. Operation Design Process

```
┌─────────────────────────────────────────────────────────┐
│               OPERATION DESIGN PROCESS                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Identify the Need                                   │
│     • What specific action needs to be performed?       │
│     • What is the expected outcome?                     │
│                                                         │
│  2. Choose Core Operation                               │
│     • Which primary operation category best fits?       │
│     • What specific action within that category?        │
│                                                         │
│  3. Select Appropriate Modifier                         │
│     • How should the operation be qualified?            │
│     • What variant or method is needed?                 │
│                                                         │
│  4. Define Parameters                                   │
│     • What inputs control the operation?                │
│     • What settings or options are needed?              │
│                                                         │
│  5. Test and Refine                                     │
│     • Does the operation produce the expected result?   │
│     • How can it be optimized?                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 11.2. Core Operation Selection Guide

When choosing a core operation, consider these questions:

1. **Purpose**: What is the primary goal?
   - Extract information → `/extract`
   - Remove information → `/filter`
   - Change format → `/restructure` or `/format`
   - Reduce size → `/compress`
   - Analyze content → `/analyze`
   - Generate insights → `/synthesize`

2. **Scope**: What is being operated on?
   - Entire documents → `/document`
   - Conversation history → `/history`
   - Field dynamics → `/field`, `/attractor`, `/boundary`
   - Memory management → `/remember`, `/retrieve`

3. **Complexity**: How complex is the operation?
   - Simple, single action → Basic operation
   - Conditional action → `/if`
   - Multiple items → `/for.each`
   - Sequence of operations → `/pipeline`

### 11.3. Parameter Design Guidelines

Effective parameters follow these principles:

1. **Clarity**: Use descriptive parameter names
   - Good: `method="extractive_summary"`
   - Poor: `m="e"`

2. **Completeness**: Include all necessary parameters
   - Input sources: `from`, `source`, `target`
   - Control parameters: `threshold`, `method`, `style`
   - Output control: `format`, `include`, `exclude`

3. **Defaults**: Consider what happens when parameters are omitted
   - What reasonable defaults apply?
   - Which parameters are absolutely required?

4. **Types**: Use appropriate value types
   - Strings for names, methods, styles
   - Numbers for thresholds, counts, weights
   - Booleans for flags
   - Arrays for multiple values
   - Nested operations for complex parameters

### 11.4. Example Development Process

Let's walk through developing a custom operation:

**Need**: Extract key information from a meeting transcript, categorize it, and format it as structured notes.

**Step 1**: Identify the core operation and modifier
- Primary action is extraction → `/extract`
- Specific variant is meeting information → `/extract.meeting_notes`

**Step 2**: Define the parameters
```
/extract.meeting_notes{
    transcript="[Meeting transcript text]",
    categories=["decisions", "action_items", "discussions", "follow_ups"],
    participants=["Alice", "Bob", "Charlie"],
    format="structured"
}
```

**Step 3**: Refine with additional control parameters
