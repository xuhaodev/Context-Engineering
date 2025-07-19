# Information Theory: Mathematical Foundations for Context Engineering

*Maximizing Information Content and Retrieval Effectiveness*

## The Information-Theoretic Foundation of Context Engineering

The survey reveals that Context Engineering fundamentally operates on information-theoretic principles. The core insight is that optimal context construction requires maximizing mutual information between context components and target outputs, rather than simply maximizing semantic similarity.

### Core Information-Theoretic Formulation

The foundational retrieval optimization problem from the survey:

```math
\text{Retrieve}^* = \arg\max_{\text{Retrieve}} I(Y^*; c_{\text{know}}|c_{\text{query}})
```

Where 
```math
$I(Y^*; c_{\text{know}}|c_{\text{query}})$
```
represents conditional mutual information between the target output
```math
$Y^*$
```

and retrieved knowledge
```math
$c_{\text{know}}$ given the query $c_{\text{query}}$.
```
This principle extends beyond retrieval to all context engineering operations:

```math
\begin{align}
\text{Assembly}^* &= \arg\max_A I(Y^*; A(c_1, ..., c_n)|c_{\text{query}}) \\
\text{Compression}^* &= \arg\max_{\text{Compress}} I(Y^*; \text{Compress}(C)) - \lambda \cdot ||\text{Compress}(C)||_1 \\
\text{Selection}^* &= \arg\max_{\text{Select}} I(Y^*; \text{Select}(C)) \text{ s.t. } |\text{Select}(C)| \leq L_{\max}
\end{align}
```

## Fundamental Information Measures

### 1. Entropy and Information Content

**Shannon Entropy** measures the information content of context components:

```math
H(C) = -\sum_{c \in \mathcal{C}} P(c) \log P(c)
```

**Conditional Entropy** quantifies remaining uncertainty:

```math
H(Y|C) = -\sum_{y,c} P(y,c) \log P(y|c)
```

**Implementation Framework**:

```python
import numpy as np
from scipy.stats import entropy
from collections import Counter
import torch
from typing import List, Dict, Tuple

class InformationMeasures:
    """Implementation of core information-theoretic measures for context engineering"""
    
    def __init__(self, vocabulary_size: int = 50000):
        self.vocab_size = vocabulary_size
        
    def shannon_entropy(self, token_sequence: List[int]) -> float:
        """
        Calculate Shannon entropy of token sequence
        H(X) = -∑ P(x) log P(x)
        """
        if not token_sequence:
            return 0.0
        
        # Count token frequencies
        token_counts = Counter(token_sequence)
        total_tokens = len(token_sequence)
        
        # Calculate probabilities
        probabilities = [count / total_tokens for count in token_counts.values()]
        
        # Calculate Shannon entropy
        return entropy(probabilities, base=2)
    
    def conditional_entropy(self, 
                          target_sequence: List[int], 
                          context_sequence: List[int],
                          window_size: int = 10) -> float:
        """
        Calculate conditional entropy H(Y|C)
        Measures uncertainty in target given context
        """
        if len(target_sequence) != len(context_sequence):
            raise ValueError("Target and context sequences must have same length")
        
        # Build conditional probability distribution
        context_target_pairs = {}
        context_counts = Counter()
        
        for i in range(len(target_sequence)):
            # Extract context window
            start_idx = max(0, i - window_size)
            context_window = tuple(context_sequence[start_idx:i+1])
            target_token = target_sequence[i]
            
            # Count occurrences
            if context_window not in context_target_pairs:
                context_target_pairs[context_window] = Counter()
            
            context_target_pairs[context_window][target_token] += 1
            context_counts[context_window] += 1
        
        # Calculate conditional entropy
        conditional_entropy_value = 0.0
        total_count = sum(context_counts.values())
        
        for context_window, target_counter in context_target_pairs.items():
            context_prob = context_counts[context_window] / total_count
            
            # Calculate H(Y|context_window)
            context_total = sum(target_counter.values())
            target_probs = [count / context_total for count in target_counter.values()]
            context_conditional_entropy = entropy(target_probs, base=2)
            
            conditional_entropy_value += context_prob * context_conditional_entropy
        
        return conditional_entropy_value
    
    def mutual_information(self, 
                          sequence_x: List[int], 
                          sequence_y: List[int]) -> float:
        """
        Calculate mutual information I(X;Y) = H(X) - H(X|Y)
        """
        h_x = self.shannon_entropy(sequence_x)
        h_x_given_y = self.conditional_entropy(sequence_x, sequence_y)
        
        return h_x - h_x_given_y
    
    def conditional_mutual_information(self,
                                     target_sequence: List[int],
                                     context_sequence: List[int], 
                                     condition_sequence: List[int]) -> float:
        """
        Calculate conditional mutual information I(Y;C|Q)
        Core metric for context engineering optimization
        """
        # I(Y;C|Q) = H(Y|Q) - H(Y|C,Q)
        h_y_given_q = self.conditional_entropy(target_sequence, condition_sequence)
        
        # Combine context and condition sequences
        combined_sequence = [
            (c, q) for c, q in zip(context_sequence, condition_sequence)
        ]
        combined_tokens = [hash(pair) % self.vocab_size for pair in combined_sequence]
        
        h_y_given_cq = self.conditional_entropy(target_sequence, combined_tokens)
        
        return h_y_given_q - h_y_given_cq
```

### 2. Mutual Information Optimization for Retrieval

The survey emphasizes that optimal retrieval maximizes mutual information rather than similarity:

```python
class OptimalRetrieval:
    """Information-theoretic retrieval optimization"""
    
    def __init__(self, 
                 knowledge_base: List[str],
                 embedding_model,
                 tokenizer):
        self.knowledge_base = knowledge_base
        self.embed_model = embedding_model
        self.tokenizer = tokenizer
        self.info_measures = InformationMeasures()
        
        # Pre-compute embeddings and token sequences
        self.doc_embeddings = [self.embed_model(doc) for doc in knowledge_base]
        self.doc_tokens = [self.tokenizer.encode(doc) for doc in knowledge_base]
    
    def retrieve_maximum_mutual_info(self,
                                   query: str,
                                   target_examples: List[str],
                                   k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve documents maximizing I(Y*; c_know|c_query)
        """
        query_tokens = self.tokenizer.encode(query)
        target_token_sequences = [self.tokenizer.encode(example) for example in target_examples]
        
        # Calculate mutual information for each document
        mi_scores = []
        for i, doc in enumerate(self.knowledge_base):
            doc_tokens = self.doc_tokens[i]
            
            # Calculate conditional mutual information
            mi_score = self._calculate_doc_mutual_info(
                doc_tokens, query_tokens, target_token_sequences
            )
            
            mi_scores.append((doc, mi_score))
        
        # Sort by mutual information and return top-k
        mi_scores.sort(key=lambda x: x[1], reverse=True)
        return mi_scores[:k]
    
    def _calculate_doc_mutual_info(self,
                                 doc_tokens: List[int],
                                 query_tokens: List[int],
                                 target_token_sequences: List[List[int]]) -> float:
        """
        Calculate I(Y*; doc|query) across multiple target examples
        """
        total_mi = 0.0
        
        for target_tokens in target_token_sequences:
            # Ensure sequences have compatible length for analysis
            min_length = min(len(doc_tokens), len(target_tokens), len(query_tokens))
            
            if min_length < 5:  # Skip very short sequences
                continue
            
            doc_sample = doc_tokens[:min_length]
            target_sample = target_tokens[:min_length]
            query_sample = query_tokens[:min_length] if len(query_tokens) >= min_length else query_tokens * (min_length // len(query_tokens) + 1)
            query_sample = query_sample[:min_length]
            
            # Calculate conditional mutual information
            mi = self.info_measures.conditional_mutual_information(
                target_sample, doc_sample, query_sample
            )
            total_mi += mi
        
        return total_mi / len(target_token_sequences) if target_token_sequences else 0.0
```

## Advanced Information-Theoretic Techniques

### 1. Attention-Based Information Flow Analysis

Modern transformer architectures process information through attention mechanisms. We can analyze information flow using attention patterns:

```python
import torch
import torch.nn.functional as F
from typing import Optional

class AttentionInformationAnalysis:
    """Analyze information flow through attention mechanisms"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def extract_attention_patterns(self, 
                                 text: str, 
                                 layer_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """Extract attention patterns from specified layers"""
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(tokens, output_attentions=True)
            attentions = outputs.attentions
        
        if layer_indices is None:
            layer_indices = list(range(len(attentions)))
        
        extracted_attentions = {}
        for layer_idx in layer_indices:
            extracted_attentions[f'layer_{layer_idx}'] = attentions[layer_idx]
        
        return extracted_attentions
    
    def calculate_attention_entropy(self, 
                                  attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy of attention distributions
        High entropy = diffuse attention, Low entropy = focused attention
        """
        # attention_weights shape: [batch, heads, seq_len, seq_len]
        
        # Calculate entropy for each attention head
        attention_probs = F.softmax(attention_weights, dim=-1)
        log_probs = torch.log(attention_probs + 1e-12)  # Add small epsilon for numerical stability
        entropy = -torch.sum(attention_probs * log_probs, dim=-1)
        
        return entropy
    
    def analyze_information_flow(self, 
                               context: str,
                               query: str) -> Dict[str, torch.Tensor]:
        """
        Analyze how information flows from context to query through attention
        """
        # Combine context and query
        combined_text = f"{context} {self.tokenizer.sep_token} {query}"
        
        # Extract attention patterns
        attention_patterns = self.extract_attention_patterns(combined_text)
        
        # Analyze information flow
        analysis_results = {}
        
        for layer_name, attention_weights in attention_patterns.items():
            # Calculate entropy
            entropy = self.calculate_attention_entropy(attention_weights)
            analysis_results[f'{layer_name}_entropy'] = entropy
            
            # Calculate information flow from context to query
            seq_len = attention_weights.size(-1)
            context_len = len(self.tokenizer.encode(context))
            
            # Flow from context tokens to query tokens
            context_to_query_flow = attention_weights[:, :, context_len:, :context_len]
            analysis_results[f'{layer_name}_context_to_query_flow'] = context_to_query_flow.mean()
        
        return analysis_results
    
    def optimize_context_for_attention_flow(self,
                                          candidate_contexts: List[str],
                                          query: str) -> Tuple[str, float]:
        """
        Select context that maximizes information flow to query
        """
        best_context = None
        best_flow_score = -float('inf')
        
        for context in candidate_contexts:
            analysis = self.analyze_information_flow(context, query)
            
            # Calculate composite flow score
            flow_score = 0.0
            for key, value in analysis.items():
                if 'context_to_query_flow' in key:
                    flow_score += float(value)
            
            if flow_score > best_flow_score:
                best_flow_score = flow_score
                best_context = context
        
        return best_context, best_flow_score
```

### 2. Information-Theoretic Context Compression

The survey identifies context compression as a critical component. We implement compression that preserves maximum information:

```python
class InformationPreservingCompression:
    """Compress context while preserving maximum information content"""
    
    def __init__(self, 
                 compression_model,
                 information_measures: InformationMeasures):
        self.compression_model = compression_model
        self.info_measures = information_measures
        
    def compress_with_information_preservation(self,
                                             context: str,
                                             target_length: int,
                                             query: str) -> str:
        """
        Compress context to target length while maximizing I(Y*; compressed_context|query)
        """
        # Tokenize inputs
        context_tokens = self.tokenizer.encode(context)
        query_tokens = self.tokenizer.encode(query)
        
        if len(context_tokens) <= target_length:
            return context
        
        # Use sliding window approach to find optimal compression
        best_compression = None
        best_information_score = -float('inf')
        
        # Try different compression strategies
        strategies = [
            self._extract_high_information_sentences,
            self._preserve_query_relevant_segments,
            self._maintain_structural_coherence
        ]
        
        for strategy in strategies:
            compressed_context = strategy(context, target_length, query)
            
            # Evaluate information preservation
            info_score = self._evaluate_information_preservation(
                compressed_context, context, query
            )
            
            if info_score > best_information_score:
                best_information_score = info_score
                best_compression = compressed_context
        
        return best_compression
    
    def _extract_high_information_sentences(self,
                                          context: str,
                                          target_length: int,
                                          query: str) -> str:
        """Extract sentences with highest information content"""
        sentences = context.split('.')
        sentence_scores = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            # Calculate information content
            sentence_tokens = self.tokenizer.encode(sentence)
            info_content = self.info_measures.shannon_entropy(sentence_tokens)
            
            # Calculate relevance to query
            query_tokens = self.tokenizer.encode(query)
            relevance = self.info_measures.mutual_information(
                sentence_tokens[:min(len(sentence_tokens), len(query_tokens))],
                query_tokens[:min(len(sentence_tokens), len(query_tokens))]
            )
            
            # Combined score
            score = 0.7 * info_content + 0.3 * relevance
            sentence_scores.append((sentence, score))
        
        # Sort by score and select top sentences within target length
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_sentences = []
        current_length = 0
        
        for sentence, score in sentence_scores:
            sentence_length = len(self.tokenizer.encode(sentence))
            if current_length + sentence_length <= target_length:
                selected_sentences.append(sentence)
                current_length += sentence_length
            else:
                break
        
        return '. '.join(selected_sentences)
    
    def _preserve_query_relevant_segments(self,
                                        context: str,
                                        target_length: int,
                                        query: str) -> str:
        """Preserve segments most relevant to query"""
        # Implementation of query-relevant segment extraction
        # This would involve semantic similarity and attention analysis
        pass
    
    def _maintain_structural_coherence(self,
                                     context: str,
                                     target_length: int,
                                     query: str) -> str:
        """Maintain logical flow and structural coherence during compression"""
        # Implementation of structure-preserving compression
        # This would involve discourse analysis and coherence metrics
        pass
    
    def _evaluate_information_preservation(self,
                                         compressed_context: str,
                                         original_context: str,
                                         query: str) -> float:
        """
        Evaluate how well compression preserves information
        """
        # Tokenize all inputs
        compressed_tokens = self.tokenizer.encode(compressed_context)
        original_tokens = self.tokenizer.encode(original_context)
        query_tokens = self.tokenizer.encode(query)
        
        # Calculate information preservation metrics
        
        # 1. Entropy preservation ratio
        compressed_entropy = self.info_measures.shannon_entropy(compressed_tokens)
        original_entropy = self.info_measures.shannon_entropy(original_tokens)
        entropy_ratio = compressed_entropy / original_entropy if original_entropy > 0 else 0
        
        # 2. Query relevance preservation
        original_relevance = self.info_measures.mutual_information(
            original_tokens[:min(len(original_tokens), len(query_tokens))],
            query_tokens[:min(len(original_tokens), len(query_tokens))]
        )
        
        compressed_relevance = self.info_measures.mutual_information(
            compressed_tokens[:min(len(compressed_tokens), len(query_tokens))],
            query_tokens[:min(len(compressed_tokens), len(query_tokens))]
        )
        
        relevance_ratio = compressed_relevance / original_relevance if original_relevance > 0 else 0
        
        # 3. Compression efficiency
        compression_ratio = len(compressed_tokens) / len(original_tokens)
        
        # Combined information preservation score
        preservation_score = (
            0.4 * entropy_ratio +
            0.5 * relevance_ratio +
            0.1 * (1 - compression_ratio)  # Reward higher compression
        )
        
        return preservation_score
```

## Information Flow Visualization

### Visual Analysis of Information Dynamics

```python
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict

class InformationFlowVisualizer:
    """Visualize information flow in context engineering systems"""
    
    def __init__(self):
        self.fig_size = (12, 8)
        
    def plot_mutual_information_matrix(self,
                                     components: List[str],
                                     mi_matrix: np.ndarray,
                                     title: str = "Mutual Information Between Context Components"):
        """Plot mutual information matrix as heatmap"""
        plt.figure(figsize=self.fig_size)
        
        # Create heatmap
        sns.heatmap(mi_matrix, 
                   xticklabels=components,
                   yticklabels=components,
                   annot=True,
                   cmap='viridis',
                   center=0)
        
        plt.title(title)
        plt.xlabel('Context Components')
        plt.ylabel('Context Components')
        plt.tight_layout()
        plt.show()
    
    def plot_information_flow_graph(self,
                                  components: List[str],
                                  flow_strengths: Dict[Tuple[str, str], float],
                                  threshold: float = 0.1):
        """Plot information flow as directed graph"""
        G = nx.DiGraph()
        
        # Add nodes
        for component in components:
            G.add_node(component)
        
        # Add edges for flows above threshold
        for (source, target), strength in flow_strengths.items():
            if strength > threshold:
                G.add_edge(source, target, weight=strength)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        plt.figure(figsize=self.fig_size)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightblue',
                              node_size=3000,
                              alpha=0.7)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos,
                              width=[w * 5 for w in weights],  # Scale width by weight
                              alpha=0.6,
                              edge_color='red',
                              arrows=True,
                              arrowsize=20)
        
        plt.title("Information Flow Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_entropy_evolution(self,
                             time_steps: List[int],
                             entropy_values: List[float],
                             title: str = "Context Entropy Evolution"):
        """Plot how context entropy changes over time"""
        plt.figure(figsize=self.fig_size)
        
        plt.plot(time_steps, entropy_values, 'b-', linewidth=2, marker='o')
        plt.xlabel('Time Steps')
        plt.ylabel('Shannon Entropy (bits)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(time_steps, entropy_values, 1)
        p = np.poly1d(z)
        plt.plot(time_steps, p(time_steps), "r--", alpha=0.8, label='Trend')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
```

## Practical Exercise: Information-Theoretic Context Optimization

### Exercise 1: Optimal Retrieval Implementation

```python
def exercise_information_theoretic_retrieval():
    """
    Implement information-theoretic retrieval optimization
    
    Task: Build a retrieval system that maximizes I(Y*; retrieved_docs|query)
    rather than semantic similarity
    """
    
    class InformationOptimalRetriever:
        def __init__(self, knowledge_base: List[str]):
            self.knowledge_base = knowledge_base
            self.info_measures = InformationMeasures()
        
        def retrieve_optimal_documents(self,
                                     query: str,
                                     target_examples: List[str],
                                     k: int = 5) -> List[str]:
            """
            Retrieve k documents maximizing conditional mutual information
            
            Students implement:
            1. Tokenization and encoding
            2. Conditional mutual information calculation
            3. Ranking and selection
            """
            # Exercise implementation here
            pass
        
        def evaluate_retrieval_quality(self,
                                     retrieved_docs: List[str],
                                     query: str,
                                     ground_truth: List[str]) -> Dict[str, float]:
            """
            Evaluate retrieval using information-theoretic metrics
            
            Students implement:
            1. Information content analysis
            2. Relevance measurement
            3. Diversity assessment
            """
            # Exercise implementation here
            pass
    
    print("Exercise: Implement information-theoretic retrieval optimization")
```

### Exercise 2: Context Compression with Information Preservation

```python
def exercise_information_preserving_compression():
    """
    Implement context compression that preserves maximum information
    
    Task: Compress context while maintaining I(Y*; compressed_context|query)
    """
    
    class InformationPreservingCompressor:
        def __init__(self):
            self.info_measures = InformationMeasures()
        
        def compress_context(self,
                           context: str,
                           target_ratio: float,
                           query: str) -> str:
            """
            Compress context to target ratio while preserving information
            
            Students implement:
            1. Information content analysis of segments
            2. Query relevance scoring
            3. Optimal segment selection
            """
            # Exercise implementation here
            pass
        
        def evaluate_compression_quality(self,
                                       original: str,
                                       compressed: str,
                                       query: str) -> Dict[str, float]:
            """
            Evaluate compression using information-theoretic metrics
            
            Students implement:
            1. Information preservation ratio
            2. Query relevance preservation
            3. Coherence maintenance
            """
            # Exercise implementation here
            pass
    
    print("Exercise: Implement information-preserving compression")
```

## Next Steps: From Information Theory to Bayesian Inference

This information-theoretic foundation establishes the mathematical principles for optimal context engineering. The concepts developed here—mutual information maximization, entropy preservation, and information flow analysis—form the basis for more advanced techniques in subsequent modules:

1. **Bayesian Context Inference**: Using information theory to update context beliefs
2. **Dynamic Context Assembly**: Information-guided component composition
3. **Retrieval Optimization**: Systematic application of mutual information principles
4. **Context Management**: Information-theoretic compression and storage strategies

The mathematical rigor of information theory provides the objective foundation for moving beyond heuristic approaches to systematic, measurable context engineering.

---

**Information Foundation**: Information theory provides the mathematical framework for objective optimization of context content, moving beyond subjective similarity measures to principled information maximization.

**Implementation Principle**: Every information-theoretic concept presented here has direct applications in practical context engineering systems, ensuring theoretical understanding translates to measurable improvements.

**Recursive Enhancement**: The information-theoretic framework enables self-improving systems that can measure and optimize their own information processing effectiveness, embodying the meta-recursive principles of advanced context engineering.
