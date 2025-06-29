#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Context-Engineering: Schema Design for Structured Context
========================================================

This module focuses on designing structured schemas for LLM context,
enabling more consistent, verifiable, and composable interactions.
Schema-driven contexts reduce variability, increase prompt robustness,
and create a bridge between human intent and machine processing.

Key concepts covered:
1. Basic schema patterns and structures
2. Schema validation and enforcement
3. Recursive and fractal schemas
4. Field protocols as schema-driven contexts
5. Measuring schema effectiveness

Usage:
    # In Jupyter or Colab:
    %run 06_schema_design.py
    # or
    from schema_design import JSONSchema, SchemaContext, FractalSchema
"""

import os
import re
import json
import time
import uuid
import logging
import hashlib
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, TypeVar, Set
from IPython.display import display, Markdown, HTML, JSON

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required libraries
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not found. Install with: pip install openai")

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logger.warning("jsonschema not found. Install with: pip install jsonschema")

try:
    import dotenv
    dotenv.load_dotenv()
    ENV_LOADED = True
except ImportError:
    ENV_LOADED = False
    logger.warning("python-dotenv not found. Install with: pip install python-dotenv")

# Constants
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000


# Helper Functions
# ===============

def setup_client(api_key=None, model=DEFAULT_MODEL):
    """
    Set up the API client for LLM interactions.

    Args:
        api_key: API key (if None, will look for OPENAI_API_KEY in env)
        model: Model name to use

    Returns:
        tuple: (client, model_name)
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None and not ENV_LOADED:
            logger.warning("No API key found. Set OPENAI_API_KEY env var or pass api_key param.")
    
    if OPENAI_AVAILABLE:
        client = OpenAI(api_key=api_key)
        return client, model
    else:
        logger.error("OpenAI package required. Install with: pip install openai")
        return None, model


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    Count tokens in text string using the appropriate tokenizer.

    Args:
        text: Text to tokenize
        model: Model name to use for tokenization

    Returns:
        int: Token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback for when tiktoken doesn't support the model
        logger.warning(f"Could not use tiktoken for {model}: {e}")
        # Rough approximation: 1 token ≈ 4 chars in English
        return len(text) // 4


def generate_response(
    prompt: str,
    client=None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_message: str = "You are a helpful assistant."
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a response from the LLM and return with metadata.

    Args:
        prompt: The prompt to send
        client: API client (if None, will create one)
        model: Model name
        temperature: Temperature parameter
        max_tokens: Maximum tokens to generate
        system_message: System message to use

    Returns:
        tuple: (response_text, metadata)
    """
    if client is None:
        client, model = setup_client(model=model)
        if client is None:
            return "ERROR: No API client available", {"error": "No API client"}
    
    prompt_tokens = count_tokens(prompt, model)
    system_tokens = count_tokens(system_message, model)
    
    metadata = {
        "prompt_tokens": prompt_tokens,
        "system_tokens": system_tokens,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": time.time()
    }
    
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        latency = time.time() - start_time
        
        response_text = response.choices[0].message.content
        response_tokens = count_tokens(response_text, model)
        
        metadata.update({
            "latency": latency,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + system_tokens + response_tokens,
            "token_efficiency": response_tokens / (prompt_tokens + system_tokens) if (prompt_tokens + system_tokens) > 0 else 0,
            "tokens_per_second": response_tokens / latency if latency > 0 else 0
        })
        
        return response_text, metadata
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        metadata["error"] = str(e)
        return f"ERROR: {str(e)}", metadata


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dictionary into a readable string.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        str: Formatted metrics string
    """
    # Select the most important metrics to show
    key_metrics = {
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "response_tokens": metrics.get("response_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "latency": f"{metrics.get('latency', 0):.2f}s",
        "token_efficiency": f"{metrics.get('token_efficiency', 0):.2f}"
    }
    
    return " | ".join([f"{k}: {v}" for k, v in key_metrics.items()])


def display_schema_example(
    title: str,
    schema: Dict[str, Any],
    instance: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Display a schema and an instance that conforms to it.
    
    Args:
        title: Title for the display
        schema: JSON Schema
        instance: Instance conforming to the schema
        metrics: Optional metrics to display
    """
    display(HTML(f"<h2>{title}</h2>"))
    
    # Display schema
    display(HTML("<h3>Schema</h3>"))
    display(JSON(schema))
    
    # Display instance
    display(HTML("<h3>Instance</h3>"))
    display(JSON(instance))
    
    # Display metrics if provided
    if metrics:
        display(HTML("<h3>Metrics</h3>"))
        display(Markdown(f"```\n{format_metrics(metrics)}\n```"))


# Basic Schema Classes
# ===================

class JSONSchema:
    """
    A class for creating, validating, and applying JSON Schema
    to LLM contexts.
