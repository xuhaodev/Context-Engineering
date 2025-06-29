# Context Expansion Techniques: From Prompts to Layered Context

This notebook demonstrates practical approaches to expand basic prompts into richer contexts that enhance LLM performance. We'll explore how to **strategically add context layers** while measuring their impact on token usage and output quality.

## Setup and Prerequisites

Let's first import the necessary libraries:


```python
import os
import json
import time
import tiktoken  # OpenAI's tokenizer
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union

# Load environment variables (you'll need to add your API key in a .env file)
# For OpenAI API key
import dotenv
dotenv.load_dotenv()

# Define API clients (choose one based on your preference)
USE_OPENAI = True  # Set to False to use another provider

if USE_OPENAI:
    from openai import OpenAI
    client = OpenAI()
    MODEL = "gpt-3.5-turbo"  # You can change to gpt-4 or other models
else:
    # Add alternative API client setup here
    # e.g., Anthropic, Cohere, etc.
    pass

# Token counter setup
tokenizer = tiktoken.encoding_for_model(MODEL) if USE_OPENAI else None

def count_tokens(text: str) -> int:
    """Count tokens in a string using the appropriate tokenizer."""
    if tokenizer:
        return len(tokenizer.encode(text))
    # Fallback for non-OpenAI models (rough approximation)
    return len(text.split()) * 1.3  # Rough approximation

def measure_latency(func, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time
```

## 1. Understanding Context Expansion

In the previous notebook (`01_min_prompt.ipynb`), we explored the basics of atomic prompts. Now we'll see how to strategically expand these atoms into molecules (richer context structures).

Let's define some utility functions for measuring context effectiveness:


```python
def calculate_metrics(prompt: str, response: str, latency: float) -> Dict[str, float]:
    """Calculate key metrics for a prompt-response pair."""
    prompt_tokens = count_tokens(prompt)
    response_tokens = count_tokens(response)
    
    # Simple token efficiency (response tokens / prompt tokens)
    token_efficiency = response_tokens / prompt_tokens if prompt_tokens > 0 else 0
    
    # Latency per 1k tokens
    latency_per_1k = (latency / prompt_tokens) * 1000 if prompt_tokens > 0 else 0
    
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "token_efficiency": token_efficiency,
        "latency": latency,
        "latency_per_1k": latency_per_1k
    }

def generate_response(prompt: str) -> Tuple[str, float]:
    """Generate a response from the LLM and measure latency."""
    if USE_OPENAI:
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        latency = time.time() - start_time
        return response.choices[0].message.content, latency
    else:
        # Add your alternative API call here
        pass
```

## 2. Experiment: Context Expansion Techniques

Let's examine different techniques to expand a basic prompt, measuring the impact of each expansion:


```python
# Base prompt (atom)
base_prompt = "Write a paragraph about climate change."

# Expanded prompt variations (molecules)
expanded_prompts = {
    "base": base_prompt,
    
    "with_role": """You are an environmental scientist with expertise in climate systems. 
Write a paragraph about climate change.""",
    
    "with_examples": """Write a paragraph about climate change.

Example 1:
Climate change refers to long-term shifts in temperatures and weather patterns. Human activities have been the main driver of climate change since the 1800s, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases.

Example 2:
Global climate change is evident in the increasing frequency of extreme weather events, rising sea levels, and shifting wildlife populations. Scientific consensus points to human activity as the primary cause.""",
    
    "with_constraints": """Write a paragraph about climate change.
- Include at least one scientific fact with numbers
- Mention both causes and effects
- End with a call to action
- Keep the tone informative but accessible""",
    
    "with_audience": """Write a paragraph about climate change for high school students who are
just beginning to learn about environmental science. Use clear explanations 
and relatable examples.""",
    
    "comprehensive": """You are an environmental scientist with expertise in climate systems.

Write a paragraph about climate change for high school students who are
just beginning to learn about environmental science. Use clear explanations 
and relatable examples.

Guidelines:
- Include at least one scientific fact with numbers
- Mention both causes and effects
- End with a call to action
- Keep the tone informative but accessible

Example of tone and structure:
"Ocean acidification occurs when seawater absorbs CO2 from the atmosphere, causing pH levels to drop. Since the Industrial Revolution, ocean pH has decreased by 0.1 units, representing a 30% increase in acidity. This affects marine life, particularly shellfish and coral reefs, as it impairs their ability to form shells and skeletons. Scientists predict that if emissions continue at current rates, ocean acidity could increase by 150% by 2100, devastating marine ecosystems. By reducing our carbon footprint through simple actions like using public transportation, we can help protect these vital ocean habitats."
"""
}

# Run experiments
results = {}
responses = {}

for name, prompt in expanded_prompts.items():
    print(f"Testing prompt: {name}")
    response, latency = generate_response(prompt)
    responses[name] = response
    metrics = calculate_metrics(prompt, response, latency)
    results[name] = metrics
    print(f"  Prompt tokens: {metrics['prompt_tokens']}")
    print(f"  Response tokens: {metrics['response_tokens']}")
    print(f"  Latency: {metrics['latency']:.2f}s")
    print("-" * 40)
```

## 3. Visualizing and Analyzing Results


```python
# Prepare data for visualization
prompt_types = list(results.keys())
prompt_tokens = [results[k]['prompt_tokens'] for k in prompt_types]
response_tokens = [results[k]['response_tokens'] for k in prompt_types]
latencies = [results[k]['latency'] for k in prompt_types]

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Token Usage
axes[0, 0].bar(prompt_types, prompt_tokens, label='Prompt Tokens', alpha=0.7, color='blue')
axes[0, 0].bar(prompt_types, response_tokens, bottom=prompt_tokens, label='Response Tokens', alpha=0.7, color='green')
axes[0, 0].set_title('Token Usage by Prompt Type')
axes[0, 0].set_ylabel('Number of Tokens')
axes[0, 0].legend()
plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

# Plot 2: Token Efficiency (Response Tokens / Prompt Tokens)
token_efficiency = [results[k]['token_efficiency'] for k in prompt_types]
axes[0, 1].bar(prompt_types, token_efficiency, color='purple', alpha=0.7)
axes[0, 1].set_title('Token Efficiency (Response/Prompt)')
axes[0, 1].set_ylabel('Efficiency Ratio')
plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

# Plot 3: Latency
axes[1, 0].bar(prompt_types, latencies, color='red', alpha=0.7)
axes[1, 0].set_title('Response Latency')
axes[1, 0].set_ylabel('Seconds')
plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

# Plot 4: Latency per 1k tokens
latency_per_1k = [results[k]['latency_per_1k'] for k in prompt_types]
axes[1, 1].bar(prompt_types, latency_per_1k, color='orange', alpha=0.7)
axes[1, 1].set_title('Latency per 1k Tokens')
axes[1, 1].set_ylabel('Seconds per 1k Tokens')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
```

## 4. Qualitative Analysis

Let's examine the actual responses to assess quality differences:


```python
for name, response in responses.items():
    print(f"=== Response for {name} prompt ===")
    print(response)
    print("\n" + "=" * 80 + "\n")
```

## 5. Context Expansion Patterns

Based on our experiments, we can identify several effective context expansion patterns:

1. **Role Assignment**: Defining who the model should act as
2. **Few-Shot Examples**: Providing sample outputs to guide response format and quality
3. **Constraint Definition**: Setting boundaries and requirements for the response
4. **Audience Specification**: Clarifying who the response is intended for
5. **Comprehensive Context**: Combining multiple context elements strategically

Let's formalize these patterns into a reusable template:


```python
def create_expanded_context(
    base_prompt: str, 
    role: Optional[str] = None,
    examples: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    audience: Optional[str] = None,
    tone: Optional[str] = None,
    output_format: Optional[str] = None
) -> str:
    """
    Create an expanded context from a base prompt with optional components.
    
    Args:
        base_prompt: The core instruction or question
        role: Who the model should act as
        examples: List of example outputs to guide the model
        constraints: List of requirements or boundaries
        audience: Who the output is intended for
        tone: Desired tone of the response
        output_format: Specific format requirements
        
    Returns:
        Expanded context as a string
    """
    context_parts = []
    
    # Add role if provided
    if role:
        context_parts.append(f"You are {role}.")
    
    # Add base prompt
    context_parts.append(base_prompt)
    
    # Add audience if provided
    if audience:
        context_parts.append(f"Your response should be suitable for {audience}.")
    
    # Add tone if provided
    if tone:
        context_parts.append(f"Use a {tone} tone in your response.")
    
    # Add output format if provided
    if output_format:
        context_parts.append(f"Format your response as {output_format}.")
    
    # Add constraints if provided
    if constraints and len(constraints) > 0:
        context_parts.append("Requirements:")
        for constraint in constraints:
            context_parts.append(f"- {constraint}")
    
    # Add examples if provided
    if examples and len(examples) > 0:
        context_parts.append("Examples:")
        for i, example in enumerate(examples, 1):
            context_parts.append(f"Example {i}:\n{example}")
    
    # Join all parts with appropriate spacing
    expanded_context = "\n\n".join(context_parts)
    
    return expanded_context
```

Let's test our template with a new prompt:


```python
# Test our template
new_base_prompt = "Explain how photosynthesis works."

new_expanded_context = create_expanded_context(
    base_prompt=new_base_prompt,
    role="a biology teacher with 15 years of experience",
    audience="middle school students",
    tone="enthusiastic and educational",
    constraints=[
        "Use a plant-to-factory analogy",
        "Mention the role of chlorophyll",
        "Explain the importance for Earth's ecosystem",
        "Keep it under 200 words"
    ],
    examples=[
        "Photosynthesis is like a tiny factory inside plants. Just as a factory needs raw materials, energy, and workers to make products, plants need carbon dioxide, water, sunlight, and chlorophyll to make glucose (sugar) and oxygen. The sunlight is the energy source, chlorophyll molecules are the workers that capture this energy, while carbon dioxide and water are the raw materials. The factory's products are glucose, which the plant uses for growth and energy storage, and oxygen, which is released into the air for animals like us to breathe. This process is essential for life on Earth because it provides the oxygen we need and removes carbon dioxide from the atmosphere."
    ]
)

print("Template-generated expanded context:")
print("-" * 80)
print(new_expanded_context)
print("-" * 80)
print(f"Token count: {count_tokens(new_expanded_context)}")

# Generate a response using our expanded context
response, latency = generate_response(new_expanded_context)
metrics = calculate_metrics(new_expanded_context, response, latency)

print("\nResponse:")
print("-" * 80)
print(response)
print("-" * 80)
print(f"Response tokens: {metrics['response_tokens']}")
print(f"Latency: {metrics['latency']:.2f}s")
```

## 6. Advanced Context Expansion: Layer Optimization

In real-world applications, we need to find the optimal balance between context richness and token efficiency. Let's experiment with a systematic approach to context layer optimization:


```python
def test_layered_contexts(base_prompt: str, context_layers: Dict[str, str]) -> Dict[str, Dict]:
    """
    Test different combinations of context layers to find optimal configurations.
    
    Args:
        base_prompt: Core instruction
        context_layers: Dictionary of layer name -> layer content
        
    Returns:
        Results dictionary with metrics for each tested configuration
    """
    layer_results = {}
    
    # Test base prompt alone
    print("Testing base prompt...")
    base_response, base_latency = generate_response(base_prompt)
    layer_results["base"] = {
        "prompt": base_prompt,
        "response": base_response,
        **calculate_metrics(base_prompt, base_response, base_latency)
    }
    
    # Test each layer individually added to base
    for layer_name, layer_content in context_layers.items():
        combined_prompt = f"{base_prompt}\n\n{layer_content}"
        print(f"Testing base + {layer_name}...")
        response, latency = generate_response(combined_prompt)
        layer_results[f"base+{layer_name}"] = {
            "prompt": combined_prompt,
            "response": response,
            **calculate_metrics(combined_prompt, response, latency)
        }
    
    # Test all layers combined
    all_layers = "\n\n".join(context_layers.values())
    full_prompt = f"{base_prompt}\n\n{all_layers}"
    print("Testing all layers combined...")
    full_response, full_latency = generate_response(full_prompt)
    layer_results["all_layers"] = {
        "prompt": full_prompt,
        "response": full_response,
        **calculate_metrics(full_prompt, full_response, full_latency)
    }
    
    return layer_results

# Define a base prompt and separate context layers
layer_test_prompt = "Write code to implement a simple weather app."

context_layers = {
    "role": "You are a senior software engineer with expertise in full-stack development and UI/UX design.",
    
    "requirements": """Requirements:
- The app should show current temperature, conditions, and forecast for the next 3 days
- It should allow users to search for weather by city name
- It should have a clean, responsive interface
- The app should handle error states gracefully""",
    
    "tech_stack": """Technical specifications:
- Use HTML, CSS, and vanilla JavaScript (no frameworks)
- Use the OpenWeatherMap API for weather data
- All code should be well-commented and follow best practices
- Include both the HTML structure and JavaScript functionality""",
    
    "example": """Example structure (but improve upon this):
```html
<!DOCTYPE html>
<html>
<head>
    <title>Weather App</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Weather App</h1>
        <div class="search">
            <input type="text" placeholder="Enter city name">
            <button>Search</button>
        </div>
        <div class="weather-display">
            <!-- Weather data will be displayed here -->
        </div>
    </div>
    <script src="app.js"></script>
</body>
</html>
```"""
}

# Run the layer optimization test
layer_test_results = test_layered_contexts(layer_test_prompt, context_layers)
```

Let's visualize the results of our layer optimization test:


```python
# Extract data for visualization
config_names = list(layer_test_results.keys())
prompt_sizes = [layer_test_results[k]['prompt_tokens'] for k in config_names]
response_sizes = [layer_test_results[k]['response_tokens'] for k in config_names]
efficiencies = [layer_test_results[k]['token_efficiency'] for k in config_names]

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Token usage by configuration
axes[0].bar(config_names, prompt_sizes, label='Prompt Tokens', alpha=0.7, color='blue')
axes[0].bar(config_names, response_sizes, label='Response Tokens', alpha=0.7, color='green')
axes[0].set_title('Token Usage by Context Configuration')
axes[0].set_ylabel('Number of Tokens')
axes[0].legend()
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')

# Plot 2: Token efficiency by configuration
axes[1].bar(config_names, efficiencies, color='purple', alpha=0.7)
axes[1].set_title('Token Efficiency by Context Configuration')
axes[1].set_ylabel('Efficiency Ratio (Response/Prompt)')
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Identify the most efficient configuration
most_efficient = max(config_names, key=lambda x: layer_test_results[x]['token_efficiency'])
print(f"Most token-efficient configuration: {most_efficient}")
print(f"Efficiency ratio: {layer_test_results[most_efficient]['token_efficiency']:.2f}")
```

## 7. Context Compression Techniques

As we expand context, we often need to optimize for token usage. Let's explore some techniques for context compression:


```python
def compress_context(context: str, technique: str = 'summarize') -> str:
    """
    Apply different compression techniques to reduce token usage while preserving meaning.
    
    Args:
        context: The context to compress
        technique: Compression technique to use (summarize, keywords, bullet)
        
    Returns:
        Compressed context
    """
    if technique == 'summarize':
        # Use the LLM to summarize the context
        prompt = f"""Summarize the following context in a concise way that preserves all key information
but uses fewer words. Focus on essential instructions and details:

{context}"""
        compressed, _ = generate_response(prompt)
        return compressed
    
    elif technique == 'keywords':
        # Extract key terms and phrases
        prompt = f"""Extract the most important keywords, phrases, and instructions from this context:

{context}

Format your response as a comma-separated list of essential terms and short phrases."""
        keywords, _ = generate_response(prompt)
        return keywords
    
    elif technique == 'bullet':
        # Convert to bullet points
        prompt = f"""Convert this context into a concise, structured list of bullet points that
captures all essential information with minimal words:

{context}"""
        bullets, _ = generate_response(prompt)
        return bullets
    
    else:
        return context  # No compression

# Test compression on our comprehensive example
original_context = expanded_prompts["comprehensive"]
print(f"Original context token count: {count_tokens(original_context)}")

for technique in ['summarize', 'keywords', 'bullet']:
    compressed = compress_context(original_context, technique)
    compression_ratio = count_tokens(compressed) / count_tokens(original_context)
    print(f"\n{technique.upper()} COMPRESSION:")
    print("-" * 80)
    print(compressed)
    print("-" * 80)
    print(f"Compressed token count: {count_tokens(compressed)}")
    print(f"Compression ratio: {compression_ratio:.2f} (lower is better)")
```

## 8. Context Pruning: Deleting What Doesn't Help

Sometimes adding context layers doesn't improve performance. Let's implement a method to measure and prune unnecessary context:


```python
def evaluate_response_quality(prompt: str, response: str, criteria: List[str]) -> float:
    """
    Use the
