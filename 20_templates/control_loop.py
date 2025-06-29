"""
Context-Engineering Control Loop Template
----------------------------------------

This template provides a flexible control loop implementation for orchestrating
context-based interactions with language models. It allows for:

1. Multi-step reasoning processes
2. State tracking across interactions
3. Dynamic context management
4. Outcome evaluation and refinement

Usage:
    control_loop = ControlLoop(
        model="gpt-4",
        initial_context={"goal": "Solve this math problem step by step"},
        max_iterations=5
    )
    result = control_loop.run(input_data="What is the square root of 144?")
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("control_loop")

# ------------------------------------------------------------------------------
# Model Interface
# ------------------------------------------------------------------------------

class ModelInterface(ABC):
    """Abstract base class for language model interfaces."""
    
    @abstractmethod
    def generate(self, context: str, max_tokens: int = 1000) -> str:
        """Generate a response from the model given a context."""
        pass

class OpenAIInterface(ModelInterface):
    """OpenAI API interface for language models."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the OpenAI interface.
        
        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (optional if set in environment)
        """
        try:
            import openai
            self.openai = openai
            if api_key:
                openai.api_key = api_key
            self.model_name = model_name
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
    
    def generate(self, context: str, max_tokens: int = 1000) -> str:
        """Generate a response using the OpenAI API."""
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": context}],
                max_tokens=max_tokens,
                n=1,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class AnthropicInterface(ModelInterface):
    """Anthropic API interface for Claude models."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the Anthropic interface.
        
        Args:
            model_name: Name of the Anthropic model to use
            api_key: Anthropic API key (optional if set in environment)
        """
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name = model_name
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with 'pip install anthropic'")
    
    def generate(self, context: str, max_tokens: int = 1000) -> str:
        """Generate a response using the Anthropic API."""
        try:
            response = self.client.completion(
                model=self.model_name,
                prompt=f"\n\nHuman: {context}\n\nAssistant:",
                max_tokens_to_sample=max_tokens,
                temperature=0.7,
            )
            return response.completion
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

# ------------------------------------------------------------------------------
# Context Management
# ------------------------------------------------------------------------------

class ContextManager:
    """Manages the context for language model interactions."""
    
    def __init__(self, 
                 initial_context: Dict[str, Any] = None, 
                 max_tokens: int = 4000,
                 reserved_tokens: int = 1000):
        """
        Initialize the context manager.
        
        Args:
            initial_context: Initial context dictionary
            max_tokens: Maximum number of tokens in context
            reserved_tokens: Tokens reserved for model response
        """
        self.context = initial_context or {}
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.history: List[Dict[str, Any]] = []
    
    def update(self, key: str, value: Any) -> None:
        """Update a specific context element."""
        self.context[key] = value
    
    def get_context_str(self, template: Optional[str] = None) -> str:
        """
        Get the formatted context string based on template or default format.
        
        Args:
            template: Optional template string with {placeholders}
            
        Returns:
            Formatted context string
        """
        if template:
            try:
                return template.format(**self.context)
            except KeyError as e:
                logger.warning(f"Template key error: {e}. Using default format.")
                # Fall back to default formatting
        
        # Default formatting
        parts = []
        
        # Add system instructions if present
        if "system" in self.context:
            parts.append(f"# Instructions\n{self.context['system']}\n\n")
        
        # Add goal if present
        if "goal" in self.context:
            parts.append(f"# Goal\n{self.context['goal']}\n\n")
        
        # Add context elements
        for key, value in self.context.items():
            if key not in ["system", "goal", "history", "current_input"]:
                parts.append(f"# {key.replace('_', ' ').title()}\n{value}\n\n")
        
        # Add history if present
        if "history" in self.context and self.context["history"]:
            parts.append("# Previous Steps\n")
            for i, entry in enumerate(self.context["history"]):
                parts.append(f"Step {i+1}: {entry}\n")
            parts.append("\n")
        
        # Add current input if present
        if "current_input" in self.context:
            parts.append(f"# Current Task\n{self.context['current_input']}\n\n")
        
        # Ensure the context isn't too long
        context_str = "".join(parts)
        self._prune_if_needed(context_str)
        
        return context_str
    
    def _prune_if_needed(self, context_str: str) -> str:
        """
        Prune context if it exceeds the maximum token limit.
        
        Args:
            context_str: The current context string
            
        Returns:
            Pruned context string
        """
        # Estimate token count (rough approximation)
        estimated_tokens = len(context_str.split())
        
        if estimated_tokens > (self.max_tokens - self.reserved_tokens):
            logger.warning(f"Context too long ({estimated_tokens} words). Pruning...")
            
            # Simple pruning strategy: remove oldest history entries
            if "history" in self.context and self.context["history"]:
                self.context["history"] = self.context["history"][1:]
                logger.info("Removed oldest history entry")
                
                # Recursively check if we need to prune more
                return self._prune_if_needed(self.get_context_str())
        
        return context_str
    
    def add_to_history(self, entry: Any) -> None:
        """Add an entry to the interaction history."""
        if "history" not in self.context:
            self.context["history"] = []
        
        self.context["history"].append(entry)
        self.history.append({"timestamp": time.time(), "entry": entry})
    
    def clear_history(self) -> None:
        """Clear the interaction history."""
        if "history" in self.context:
            self.context["history"] = []

# ------------------------------------------------------------------------------
# Evaluation Functions
# ------------------------------------------------------------------------------

class EvaluationFunction(ABC):
    """Base class for evaluation functions."""
    
    @abstractmethod
    def evaluate(self, response: str, context: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Evaluate a model response.
        
        Args:
            response: The model's response
            context: The current context dictionary
            
        Returns:
            Tuple of (success_flag, score, feedback)
        """
        pass

class SimpleKeywordEvaluator(EvaluationFunction):
    """Evaluates responses based on keyword presence."""
    
    def __init__(self, required_keywords: List[str], forbidden_keywords: List[str] = None):
        """
        Initialize the keyword evaluator.
        
        Args:
            required_keywords: List of keywords that should be present
            forbidden_keywords: List of keywords that should not be present
        """
        self.required_keywords = required_keywords
        self.forbidden_keywords = forbidden_keywords or []
    
    def evaluate(self, response: str, context: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Evaluate based on keyword presence.
        
        Returns:
            Tuple of (success_flag, score, feedback)
        """
        response_lower = response.lower()
        
        # Check required keywords
        missing_keywords = [kw for kw in self.required_keywords 
                           if kw.lower() not in response_lower]
        
        # Check forbidden keywords
        present_forbidden = [kw for kw in self.forbidden_keywords 
                            if kw.lower() in response_lower]
        
        # Calculate score (0.0 to 1.0)
        if self.required_keywords:
            required_score = (len(self.required_keywords) - len(missing_keywords)) / len(self.required_keywords)
        else:
            required_score = 1.0
            
        if self.forbidden_keywords:
            forbidden_score = (len(self.forbidden_keywords) - len(present_forbidden)) / len(self.forbidden_keywords)
        else:
            forbidden_score = 1.0
            
        score = (required_score + forbidden_score) / 2.0
        success = score > 0.8  # Consider successful if score > 80%
        
        # Generate feedback
        feedback = []
        if missing_keywords:
            feedback.append(f"Missing required keywords: {', '.join(missing_keywords)}")
        if present_forbidden:
            feedback.append(f"Contains forbidden keywords: {', '.join(present_forbidden)}")
        if not feedback:
            feedback.append("Response meets keyword criteria")
            
        return success, score, "; ".join(feedback)

class PatternMatchEvaluator(EvaluationFunction):
    """Evaluates responses based on regex pattern matching."""
    
    def __init__(self, required_patterns: List[str], forbidden_patterns: List[str] = None):
        """
        Initialize the pattern evaluator.
        
        Args:
            required_patterns: List of regex patterns that should match
            forbidden_patterns: List of regex patterns that should not match
        """
        import re
        self.re = re
        self.required_patterns = [re.compile(p, re.IGNORECASE) for p in required_patterns]
        self.forbidden_patterns = [re.compile(p, re.IGNORECASE) for p in (forbidden_patterns or [])]
    
    def evaluate(self, response: str, context: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Evaluate based on pattern matching.
        
        Returns:
            Tuple of (success_flag, score, feedback)
        """
        # Check required patterns
        missing_patterns = [p.pattern for p in self.required_patterns 
                           if not p.search(response)]
        
        # Check forbidden patterns
        present_forbidden = [p.pattern for p in self.forbidden_patterns 
                            if p.search(response)]
        
        # Calculate score
        if self.required_patterns:
            required_score = (len(self.required_patterns) - len(missing_patterns)) / len(self.required_patterns)
        else:
            required_score = 1.0
            
        if self.forbidden_patterns:
            forbidden_score = (len(self.forbidden_patterns) - len(present_forbidden)) / len(self.forbidden_patterns)
        else:
            forbidden_score = 1.0
            
        score = (required_score + forbidden_score) / 2.0
        success = score > 0.8  # Consider successful if score > 80%
        
        # Generate feedback
        feedback = []
        if missing_patterns:
            feedback.append(f"Missing required patterns: {', '.join(missing_patterns)}")
        if present_forbidden:
            feedback.append(f"Contains forbidden patterns: {', '.join(present_forbidden)}")
        if not feedback:
            feedback.append("Response meets pattern criteria")
            
        return success, score, "; ".join(feedback)

class ModelEvaluator(EvaluationFunction):
    """Uses a model to evaluate another model's response."""
    
    def __init__(self, model_interface: ModelInterface, evaluation_prompt_template: str):
        """
        Initialize the model evaluator.
        
        Args:
            model_interface: ModelInterface instance for evaluation
            evaluation_prompt_template: Template for evaluation prompt
        """
        self.model = model_interface
        self.evaluation_prompt_template = evaluation_prompt_template
    
    def evaluate(self, response: str, context: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Evaluate using another model.
        
        Returns:
            Tuple of (success_flag, score, feedback)
        """
        # Create evaluation prompt
        eval_prompt = self.evaluation_prompt_template.format(
            response=response,
            **context
        )
        
        # Get evaluation from model
        try:
            eval_response = self.model.generate(eval_prompt)
            
            # Try to parse structured response (JSON)
            try:
                result = json.loads(eval_response)
                success = result.get("success", False)
                score = result.get("score", 0.0)
                feedback = result.get("feedback", "No feedback provided")
            except json.JSONDecodeError:
                # If not JSON, try to extract score and feedback heuristically
                if "score" in eval_response.lower():
                    # Try to extract score (0-10 or 0-100 scale)
                    import re
                    score_match = re.search(r"score\s*(?::|=)\s*(\d+(?:\.\d+)?)", eval_response, re.IGNORECASE)
                    if score_match:
                        raw_score = float(score_match.group(1))
                        # Normalize to 0-1 scale
                        if raw_score > 10:
                            score = raw_score / 100.0
                        else:
                            score = raw_score / 10.0
                    else:
                        score = 0.5  # Default middle score
                else:
                    score = 0.5
                
                # Simple heuristic for success based on positive language
                positive_terms = ["good", "great", "excellent", "correct", "accurate", "yes", "pass"]
                negative_terms = ["bad", "poor", "incorrect", "inaccurate", "wrong", "no", "fail"]
                
                pos_count = sum(1 for term in positive_terms if term in eval_response.lower())
                neg_count = sum(1 for term in negative_terms if term in eval_response.lower())
                
                success = pos_count > neg_count
                feedback = eval_response.strip()
            
            return success, score, feedback
            
        except Exception as e:
            logger.error(f"Evaluation model error: {e}")
            return False, 0.0, f"Evaluation failed: {str(e)}"

# ------------------------------------------------------------------------------
# Control Loop
# ------------------------------------------------------------------------------

class ControlLoop:
    """
    Main control loop for context-based LLM interactions.
    Manages the flow of information, context updates, and evaluation.
    """
    
    def __init__(self, 
                 model: Union[str, ModelInterface],
                 initial_context: Dict[str, Any] = None,
                 context_template: Optional[str] = None,
                 max_iterations: int = 5,
                 evaluators: List[EvaluationFunction] = None,
                 stop_on_success: bool = True,
                 success_threshold: float = 0.8):
        """
        Initialize the control loop.
        
        Args:
            model: Model name or ModelInterface instance
            initial_context: Initial context dictionary
            context_template: Optional template for context formatting
            max_iterations: Maximum number of iterations
            evaluators: List of EvaluationFunction instances
            stop_on_success: Whether to stop iterating on first success
            success_threshold: Threshold for considering an iteration successful
        """
        # Set up model interface
        if isinstance(model, str):
            if "gpt" in model.lower():
                self.model = OpenAIInterface(model)
            elif "claude" in model.lower():
                self.model = AnthropicInterface(model)
            else:
                raise ValueError(f"Unknown model type: {model}")
        else:
            self.model = model
        
        # Set up context manager
        self.context_manager = ContextManager(initial_context)
        self.context_template = context_template
        
        # Set up control parameters
        self.max_iterations = max_iterations
        self.evaluators = evaluators or []
        self.stop_on_success = stop_on_success
        self.success_threshold = success_threshold
        
        # Set up tracking
        self.iterations = 0
        self.results = []
    
    def add_evaluator(self, evaluator: EvaluationFunction) -> None:
        """Add an evaluation function."""
        self.evaluators.append(evaluator)
    
    def run(self, input_data: Any = None) -> Dict[str, Any]:
        """
        Run the control loop with the given input.
        
        Args:
            input_data: Input data for the loop
            
        Returns:
            Result dictionary with final response and metadata
        """
        logger.info("Starting control loop")
        self.iterations = 0
        self.results = []
        
        # Add input to context
        if input_data:
            self.context_manager.update("current_input", input_data)
        
        final_response = None
        successful = False
        
        # Main control loop
        while self.iterations < self.max_iterations:
            self.iterations += 1
            logger.info(f"Iteration {self.iterations}/{self.max_iterations}")
            
            # Get formatted context
            context_str = self.context_manager.get_context_str(self.context_template)
            
            # Generate response from model
            try:
                response = self.model.generate(context_str)
                logger.info(f"Received response ({len(response)} chars)")
            except Exception as e:
                logger.error(f"Model generation failed: {e}")
                break
                
            # Store the response
            final_response = response
            
            # Evaluate the response
            evaluation_results = []
            overall_success = True
            overall_score = 1.0
            
            for evaluator in self.evaluators:
                success, score, feedback = evaluator.evaluate(
                    response, 
                    self.context_manager.context
                )
                evaluation_results.append({
                    "evaluator": evaluator.__class__.__name__,
                    "success": success,
                    "score": score,
                    "feedback": feedback
                })
                
                # Update overall results
                overall_success = overall_success and success
                overall_score *= score  # Multiply scores for a stricter measure
            
            # Store results
            iteration_result = {
                "iteration": self.iterations,
                "response": response,
                "evaluations": evaluation_results,
                "success": overall_success,
                "score": overall_score
            }
            self.results.append(iteration_result)
            
            # Add to history
            self.context_manager.add_to_history(
                f"Response: {response}\nEvaluation: {'Success' if overall_success else 'Failure'}"
            )
            
            # Check if we should stop
            if overall_success and self.stop_on_success:
                logger.info("Stopping on successful iteration")
                successful = True
                break
                
            # Check if we've reached the maximum iterations
            if self.iterations >= self.max_iterations:
                logger.info(f"Reached maximum iterations ({self.max_iterations})")
                break
        
        # Prepare final result
        result = {
            "successful": successful,
            "iterations": self.iterations,
            "final_response": final_response,
            "detailed_results": self.results,
            "context": self.context_manager.context
        }
        
        logger.info(f"Control loop completed: {'Success' if successful else 'Failure'}")
        return result
    
    def reset(self) -> None:
        """Reset the control loop to initial state."""
        self.iterations = 0
        self.results = []
        self.context_manager.clear_history()

# ------------------------------------------------------------------------------
# Neural Field Extensions
# ------------------------------------------------------------------------------

class NeuralField:
    """
    Neural field implementation for context engineering.
    Treats context as a continuous field rather than discrete tokens.
    """
    
    def __init__(self, 
                 decay_rate: float = 0.05,
                 boundary_permeability: float = 0.8,
                 resonance_bandwidth: float = 0.6,
                 attractor_formation_threshold: float = 0.7):
        """
        Initialize the neural field.
        
        Args:
            decay_rate: Base rate of pattern decay
            boundary_permeability: How easily new information enters
            resonance_bandwidth: How broadly patterns resonate
            attractor_formation_threshold: Threshold for attractor formation
        """
        self.state = {}  # Field state
        self.attractors = {}  # Stable attractors
        self.history = []  # Field evolution history
        
        # Field properties
        self.decay_rate = decay_rate
        self.boundary_permeability = boundary_permeability
        self.resonance_bandwidth = resonance_bandwidth
        self.attractor_threshold = attractor_formation_threshold
    
    def inject(self, pattern: str, strength: float = 1.0) -> 'NeuralField':
        """
        Introduce a new pattern into the field.
        
        Args:
            pattern: The information pattern to inject
            strength: The strength of the pattern
            
        Returns:
            Self for chaining
        """
        # Apply boundary filtering
        effective_strength = strength * self.boundary_permeability
        
        # Check resonance with existing attractors
        for attractor_id, attractor in self.attractors.items():
            resonance = self._calculate_resonance(pattern, attractor['pattern'])
            if resonance > 0.2:
                # Attractor pulls pattern toward it
                pattern = self._blend_patterns(
                    pattern, 
                    attractor['pattern'],
                    blend_ratio=resonance * 0.3
                )
                # Strengthen attractor
                self.attractors[attractor_id]['strength'] += resonance * 0.1
        
        # Update field state with new pattern
        if pattern in self.state:
            self.state[pattern] += effective_strength
        else:
            self.state[pattern] = effective_strength
            
        # Record history
        self.history.append(("inject", pattern, effective_strength))
        
        # Check for attractor formation
        if pattern in self.state and self.state[pattern] > self.attractor_threshold:
            self._form_attractor(pattern)
        
        # Process resonance effects
        self._process_resonance(pattern)
        
        return self
    
    def _form_attractor(self, pattern: str) -> str:
        """
        Form a new attractor around a strong pattern.
        
        Args:
            pattern: The pattern to form an attractor around
            
        Returns:
            ID of the formed attractor
        """
        attractor_id = f"attractor_{len(self.attractors)}"
        self.attractors[attractor_id] = {
            'pattern': pattern,
            'strength': self.state[pattern],
            'formation_time': len(self.history),
            'basin_width': self.resonance_bandwidth
        }
        return attractor_id
    
    def _process_resonance(self, trigger_pattern: str) -> 'NeuralField':
        """
        Process resonance effects from a trigger pattern.
        
        Args:
            trigger_pattern: The pattern triggering resonance
            
        Returns:
            Self for chaining
        """
        # For each existing pattern, calculate resonance with trigger
        resonance_effects = {}
        for pattern, strength in self.state.items():
            if pattern != trigger_pattern:
                resonance = self._calculate_resonance(pattern, trigger_pattern)
                effect = resonance * strength * 0.2
                resonance_effects[pattern] = effect
        
        # Apply resonance effects
        for pattern, effect in resonance_effects.items():
            self.state[pattern] += effect
        
        return self
    
    def decay(self) -> 'NeuralField':
        """
        Apply natural decay to all patterns.
        
        Returns:
            Self for chaining
        """
        # Apply decay to field state
        for pattern in list(self.state.keys()):
            # Patterns that resonate with attractors decay more slowly
            attractor_protection = 0
            for attractor in self.attractors.values():
                resonance = self._calculate_resonance(pattern, attractor['pattern'])
                attractor_protection += resonance * 0.5
            
            effective_decay = self.decay_rate * (1 - min(attractor_protection, 0.9))
            self.state[pattern] *= (1 - effective_decay)
            
        # Apply minimal decay to attractors
        for attractor_id in list(self.attractors.keys()):
            self.attractors[attractor_id]['strength'] *= (1 - self.decay_rate * 0.2)
            
        # Remove patterns that have decayed below threshold
        self.state = {k: v for k, v in self.state.items() if v > 0.01}
        self.attractors = {k: v for k, v in self.attractors.items() if v['strength'] > 0.1}
        
        return self
    
    def _calculate_resonance(self, pattern1: str, pattern2: str) -> float:
        """
        Calculate resonance between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Resonance score (0.0 to 1.0)
        """
        # Simple word overlap similarity
        words1 = set(pattern1.lower().split())
        words2 = set(pattern2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = len(words1.intersection(words2))
        similarity = overlap / max(len(words1), len(words2))
        
        # Apply bandwidth modulation
        resonance = similarity * self.resonance_bandwidth
        
        return resonance
    
    def _blend_patterns(self, pattern1: str, pattern2: str, blend_ratio: float) -> str:
        """
        Blend two patterns based on ratio.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            blend_ratio: Ratio of blending (0.0 to 1.0)
            
        Returns:
            Blended pattern
        """
        # Simple concatenation with weighting indication
        return f"{pattern1} {blend_ratio:.2f}↔️ {pattern2}"
    
    def measure_field_stability(self) -> float:
        """
        Measure how stable the field is.
        
        Returns:
            Stability score (0.0 to 1.0)
        """
        if not self.attractors:
            return 0.0
        
        # Measure average attractor strength
        avg_strength = sum(a['strength'] for a in self.attractors.values()) / len(self.attractors)
        
        # Measure pattern organization around attractors
        organization = 0
        for pattern, strength in self.state.items():
            best_resonance = max(
                self._calculate_resonance(pattern, a['pattern']) 
                for a in self.attractors.values()
            ) if self.attractors else 0
            
            organization += best_resonance * strength
            
        if self.state:
            organization /= sum(self.state.values())
        else:
            organization = 0
        
        # Combine metrics
        stability = (avg_strength * 0.6) + (organization * 0.4)
        return min(1.0, stability)  # Cap at 1.0
    
    def get_context_representation(self) -> str:
        """
        Get a string representation of the current field state.
        
        Returns:
            String representation of the field
        """
        parts = []
        
        # Add attractors
        if self.attractors:
            parts.append("# Field Attractors")
            for attractor_id, attractor in self.attractors.items():
                parts.append(f"- {attractor_id} (Strength: {attractor['strength']:.2f}): {attractor['pattern'][:100]}...")
            parts.append("")
        
        # Add most active patterns
        parts.append("# Active Patterns")
        active_patterns = sorted(self.state.items(), key=lambda x: x[1], reverse=True)[:5]
        for pattern, strength in active_patterns:
            parts.append(f"- ({strength:.2f}): {pattern[:100]}...")
        
        # Add field metrics
        parts.append("")
        parts.append(f"Field Stability: {self.measure_field_stability():.2f}")
        parts.append(f"Active Patterns: {len(self.state)}")
        parts.append(f"Attractor Count: {len(self.attractors)}")
        
        return "\n".join(parts)

class NeuralFieldControlLoop(ControlLoop):
    """Control loop implementation using neural field for context management."""
    
    def __init__(self, 
                 model: Union[str, ModelInterface],
                 field_params: Dict[str, float] = None,
                 max_iterations: int = 5,
                 evaluators: List[EvaluationFunction] = None,
                 stop_on_success: bool = True,
                 success_threshold: float = 0.8):
        """
        Initialize the neural field control loop.
        
        Args:
            model: Model name or ModelInterface instance
            field_params: Parameters for the neural field
            max_iterations: Maximum number of iterations
            evaluators: List of EvaluationFunction instances
            stop_on_success: Whether to stop iterating on first success
            success_threshold: Threshold for considering an iteration successful
        """
        super().__init__(
            model=model,
            initial_context={},
            max_iterations=max_iterations,
            evaluators=evaluators,
            stop_on_success=stop_on_success,
            success_threshold=success_threshold
        )
        
        # Replace context manager with neural field
        field_params = field_params or {}
        self.field = NeuralFiel
