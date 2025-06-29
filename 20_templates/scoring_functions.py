"""
Context-Engineering Scoring Functions
------------------------------------

This module provides scoring functions to evaluate context quality and model responses
in context engineering applications. It includes metrics for:

1. Relevance - How well content relates to the query or objective
2. Coherence - How logically consistent and well-structured the content is
3. Comprehensiveness - How complete the information is
4. Conciseness - How efficiently information is presented
5. Accuracy - How factually correct the information is
6. Token Efficiency - How effectively the token budget is used
7. Field Resonance - How well content aligns with neural field patterns

Usage:
    # Score model response relevance
    relevance_score = score_relevance(response, query)
    
    # Score context coherence
    coherence_score = score_coherence(context)
    
    # Get comprehensive scoring for a response
    scores = score_response(response, query, context, reference=None)
"""

import math
import re
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("scoring_functions")

# ------------------------------------------------------------------------------
# Text Processing Utilities
# ------------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Simple tokenization function for text.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Split into tokens
    return text.split()

def count_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.
    This is a rough approximation for planning purposes.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough approximation: average token is ~4 characters
    # More accurate would be to use the specific tokenizer for your model
    return len(text) // 4

def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    
    # Remove empty sentences
    return [s.strip() for s in sentences if s.strip()]

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if not set1 or not set2:
        return 0.0
        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union

def cosine_similarity(vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector as word frequency dictionary
        vec2: Second vector as word frequency dictionary
        
    Returns:
        Cosine similarity (0.0 to 1.0)
    """
    if not vec1 or not vec2:
        return 0.0
        
    # Find common words
    common_words = set(vec1.keys()).intersection(set(vec2.keys()))
    
    # Calculate dot product
    dot_product = sum(vec1[word] * vec2[word] for word in common_words)
    
    # Calculate magnitudes
    mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    
    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
        
    return dot_product / (mag1 * mag2)

def get_word_frequency(text: str) -> Dict[str, int]:
    """
    Get word frequency dictionary from text.
    
    Args:
        text: Input text
        
    Returns:
        Word frequency dictionary
    """
    tokens = tokenize(text)
    return dict(Counter(tokens))

# ------------------------------------------------------------------------------
# Basic Scoring Functions
# ------------------------------------------------------------------------------

def score_relevance(response: str, query: str, method: str = "cosine") -> float:
    """
    Score the relevance of a response to a query.
    
    Args:
        response: Model response
        query: Original query
        method: Similarity method ("cosine" or "jaccard")
        
    Returns:
        Relevance score (0.0 to 1.0)
    """
    if not response or not query:
        return 0.0
        
    if method == "jaccard":
        # Jaccard similarity on token sets
        response_tokens = set(tokenize(response))
        query_tokens = set(tokenize(query))
        
        return jaccard_similarity(response_tokens, query_tokens)
        
    else:  # Default to cosine
        # Cosine similarity on word frequencies
        response_freq = get_word_frequency(response)
        query_freq = get_word_frequency(query)
        
        return cosine_similarity(response_freq, query_freq)

def score_coherence(text: str) -> float:
    """
    Score the coherence of text based on sentence flow and structure.
    
    Args:
        text: Input text
        
    Returns:
        Coherence score (0.0 to 1.0)
    """
    # Extract sentences
    sentences = extract_sentences(text)
    
    if len(sentences) <= 1:
        return 1.0  # Single sentence is coherent by default
        
    # Measure inter-sentence similarity
    total_similarity = 0.0
    
    for i in range(len(sentences) - 1):
        sent1 = sentences[i]
        sent2 = sentences[i + 1]
        
        # Get word sets
        words1 = set(tokenize(sent1))
        words2 = set(tokenize(sent2))
        
        # Calculate similarity
        similarity = jaccard_similarity(words1, words2)
        total_similarity += similarity
    
    # Average similarity
    avg_similarity = total_similarity / (len(sentences) - 1)
    
    # Check for transition words/phrases
    transition_words = [
        "however", "therefore", "thus", "consequently", "furthermore",
        "in addition", "moreover", "similarly", "in contrast", "nonetheless",
        "despite", "although", "because", "since", "as a result"
    ]
    
    transition_count = 0
    for sentence in sentences[1:]:  # Skip first sentence
        if any(word in sentence.lower() for word in transition_words):
            transition_count += 1
    
    transition_ratio = transition_count / (len(sentences) - 1) if len(sentences) > 1 else 0
    
    # Combine metrics (weighted average)
    coherence = (avg_similarity * 0.7) + (transition_ratio * 0.3)
    
    return coherence

def score_comprehensiveness(response: str, reference: Optional[str] = None, key_points: Optional[List[str]] = None) -> float:
    """
    Score the comprehensiveness of a response.
    
    Args:
        response: Model response
        reference: Optional reference answer
        key_points: Optional list of key points that should be covered
        
    Returns:
        Comprehensiveness score (0.0 to 1.0)
    """
    if not response:
        return 0.0
        
    # If reference is provided
    if reference:
        # Compare coverage of key terms
        response_terms = set(tokenize(response))
        reference_terms = set(tokenize(reference))
        
        # How many reference terms are covered
        coverage = len(response_terms.intersection(reference_terms)) / len(reference_terms) if reference_terms else 0
        
        return coverage
        
    # If key points are provided
    elif key_points:
        # Check how many key points are mentioned
        mentioned = 0
        for point in key_points:
            point_tokens = set(tokenize(point))
            response_tokens = set(tokenize(response))
            
            # Calculate overlap
            overlap = jaccard_similarity(point_tokens, response_tokens)
            
            if overlap > 0.3:  # Threshold for considering a point mentioned
                mentioned += 1
        
        return mentioned / len(key_points) if key_points else 0
        
    else:
        # No reference or key points, use length as a proxy
        # This is a weak proxy but better than nothing
        token_count = count_tokens(response)
        
        # Assume 150 tokens is comprehensive, scale accordingly
        return min(1.0, token_count / 150)

def score_conciseness(response: str, reference: Optional[str] = None, key_points: Optional[List[str]] = None) -> float:
    """
    Score the conciseness of a response.
    
    Args:
        response: Model response
        reference: Optional reference answer
        key_points: Optional list of key points that should be covered
        
    Returns:
        Conciseness score (0.0 to 1.0)
    """
    if not response:
        return 0.0
        
    # Get response token count
    response_tokens = count_tokens(response)
    
    # If reference is provided
    if reference:
        # Get reference token count
        reference_tokens = count_tokens(reference)
        
        # Comprehensiveness score
        comprehensiveness = score_
