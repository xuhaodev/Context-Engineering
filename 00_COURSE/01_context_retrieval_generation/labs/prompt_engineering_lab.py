#!/usr/bin/env python3
"""
Context Engineering Course - Prompt Engineering Laboratory
==========================================================

A comprehensive, research-backed implementation of advanced prompt engineering techniques
based on "A Survey of Context Engineering for Large Language Models" and formal context
engineering principles.

This laboratory provides practical implementations of:
- Chain-of-Thought (CoT) and variants
- Tree-of-Thought (ToT) reasoning
- ReAct (Reasoning + Acting) frameworks
- Self-consistency techniques
- Role-based prompting
- Meta-cognitive prompting

Usage:
    # Import the module
    from prompt_engineering_lab import *
    
    # Initialize the laboratory
    lab = PromptEngineeringLab()
    
    # Run experiments
    lab.run_chain_of_thought_experiment()
    lab.run_tree_of_thought_experiment()
    lab.run_react_experiment()

Author: Context Engineering Research Team
Version: 1.0.0
License: MIT
"""

import re
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

# Configure logging for experimental tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptingTechnique(Enum):
    """Enumeration of prompt engineering techniques."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REACT = "react"
    SELF_CONSISTENCY = "self_consistency"
    ROLE_BASED = "role_based"
    META_COGNITIVE = "meta_cognitive"
    FEW_SHOT = "few_shot"
    ZERO_SHOT = "zero_shot"

@dataclass
class PromptExperiment:
    """Data structure for tracking prompt engineering experiments."""
    technique: PromptingTechnique
    prompt_template: str
    test_query: str
    expected_outcome: Optional[str] = None
    execution_time: Optional[float] = None
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ExperimentResult:
    """Data structure for experiment results."""
    experiment: PromptExperiment
    generated_response: str
    performance_metrics: Dict[str, float]
    success: bool
    error_message: Optional[str] = None

class BasePromptFramework(ABC):
    """Abstract base class for prompt engineering frameworks."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.experiment_history: List[ExperimentResult] = []
    
    @abstractmethod
    def generate_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate a prompt using this framework."""
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response to extract structured information."""
        pass
    
    def log_experiment(self, result: ExperimentResult):
        """Log an experiment result."""
        self.experiment_history.append(result)
        logger.info(f"{self.name} experiment completed: {result.success}")

class ChainOfThoughtFramework(BasePromptFramework):
    """
    Implementation of Chain-of-Thought (CoT) prompting framework.
    
    Based on Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    """
    
    def __init__(self):
        super().__init__(
            name="Chain-of-Thought",
            description="Sequential step-by-step reasoning framework"
        )
        self.step_indicators = ["Step 1:", "Step 2:", "Step 3:", "Step 4:", "Step 5:"]
        self.conclusion_indicators = ["Therefore:", "Conclusion:", "Final Answer:"]
    
    def generate_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate a Chain-of-Thought prompt."""
        context = context or {}
        
        # Determine reasoning complexity
        complexity = context.get('complexity', 'medium')
        domain = context.get('domain', 'general')
        
        if complexity == 'high':
            cot_template = self._generate_complex_cot_template()
        elif complexity == 'low':
            cot_template = self._generate_simple_cot_template()
        else:
            cot_template = self._generate_standard_cot_template()
        
        # Add domain-specific guidance if provided
        domain_guidance = self._get_domain_guidance(domain)
        
        prompt = f"""
{domain_guidance}

Problem: {query}

{cot_template}
        """.strip()
        
        return prompt
    
    def _generate_standard_cot_template(self) -> str:
        """Generate standard CoT template."""
        return """
Let me work through this step by step:

Step 1: [Understand what is being asked]
Step 2: [Identify the key information and requirements]
Step 3: [Apply relevant knowledge or methods]
Step 4: [Work through the solution systematically]
Step 5: [Verify the answer and check for reasonableness]

Therefore: [State the final answer clearly]
        """.strip()
    
    def _generate_complex_cot_template(self) -> str:
        """Generate complex CoT template for difficult problems."""
        return """
I'll approach this complex problem systematically:

Understanding Phase:
- What exactly is being asked?
- What are the constraints and requirements?
- What domain knowledge is relevant?

Analysis Phase:
- Break down the problem into components
- Identify relationships between components
- Consider multiple approaches

Solution Phase:
- Step 1: [First logical step]
- Step 2: [Second logical step] 
- Step 3: [Third logical step]
- Step 4: [Fourth logical step]
- Step 5: [Additional steps as needed]

Verification Phase:
- Does this answer make sense?
- Have I addressed all parts of the question?
- Are there any edge cases I should consider?

Final Answer: [Complete, verified solution]
        """.strip()
    
    def _generate_simple_cot_template(self) -> str:
        """Generate simple CoT template for straightforward problems."""
        return """
Let me think through this:

1. [Identify the main question]
2. [Apply the relevant method or knowledge]
3. [Calculate or reason through the solution]

Answer: [Clear, direct answer]
        """.strip()
    
    def _get_domain_guidance(self, domain: str) -> str:
        """Get domain-specific guidance for reasoning."""
        domain_guidance = {
            'mathematical': "Use mathematical reasoning and show all calculations clearly.",
            'scientific': "Apply scientific principles and cite relevant laws or theories.",
            'logical': "Use formal logic and clearly state your premises and conclusions.",
            'creative': "Think creatively while maintaining logical structure.",
            'analytical': "Break down the problem analytically and examine each component.",
            'general': "Think systematically and show your reasoning clearly."
        }
        return domain_guidance.get(domain, domain_guidance['general'])
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse CoT response to extract reasoning steps and conclusion."""
        parsed = {
            'reasoning_steps': [],
            'conclusion': '',
            'step_count': 0,
            'has_clear_structure': False
        }
        
        # Extract reasoning steps
        step_pattern = r'Step \d+:(.+?)(?=Step \d+:|Therefore:|Conclusion:|Final Answer:|$)'
        steps = re.findall(step_pattern, response, re.DOTALL | re.IGNORECASE)
        parsed['reasoning_steps'] = [step.strip() for step in steps]
        parsed['step_count'] = len(steps)
        
        # Extract conclusion
        conclusion_pattern = r'(?:Therefore:|Conclusion:|Final Answer:|Answer:)(.+?)$'
        conclusion_match = re.search(conclusion_pattern, response, re.DOTALL | re.IGNORECASE)
        if conclusion_match:
            parsed['conclusion'] = conclusion_match.group(1).strip()
        
        # Check for clear structure
        parsed['has_clear_structure'] = (
            parsed['step_count'] > 0 and 
            len(parsed['conclusion']) > 0
        )
        
        return parsed

class TreeOfThoughtFramework(BasePromptFramework):
    """
    Implementation of Tree-of-Thought (ToT) prompting framework.
    
    Based on Yao et al. (2023) "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
    """
    
    def __init__(self):
        super().__init__(
            name="Tree-of-Thought",
            description="Parallel reasoning exploration with path evaluation"
        )
        self.max_branches = 4
        self.evaluation_criteria = ['feasibility', 'completeness', 'efficiency']
    
    def generate_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate a Tree-of-Thought prompt."""
        context = context or {}
        num_branches = context.get('num_branches', 3)
        evaluation_focus = context.get('evaluation_focus', 'comprehensive')
        
        prompt = f"""
Problem: {query}

I'll explore multiple reasoning paths simultaneously to find the best solution:

🌳 REASONING TREE EXPLORATION:

Branch A: [Approach 1 - {self._get_approach_description(1)}]
├── Step A1: [First step of this approach]
├── Step A2: [Second step of this approach]  
├── Step A3: [Third step of this approach]
└── Evaluation: [Assess this path's viability]
   • Feasibility: [How practical is this approach?]
   • Completeness: [Does this address all aspects?]
   • Efficiency: [How efficient is this solution?]

Branch B: [Approach 2 - {self._get_approach_description(2)}]
├── Step B1: [First step of this approach]
├── Step B2: [Second step of this approach]
├── Step B3: [Third step of this approach]
└── Evaluation: [Assess this path's viability]
   • Feasibility: [How practical is this approach?]
   • Completeness: [Does this address all aspects?]
   • Efficiency: [How efficient is this solution?]

Branch C: [Approach 3 - {self._get_approach_description(3)}]
├── Step C1: [First step of this approach]
├── Step C2: [Second step of this approach]
├── Step C3: [Third step of this approach]
└── Evaluation: [Assess this path's viability]
   • Feasibility: [How practical is this approach?]
   • Completeness: [Does this address all aspects?]
   • Efficiency: [How efficient is this solution?]

PATH COMPARISON AND SELECTION:
• Branch A Analysis: [Strengths and weaknesses]
• Branch B Analysis: [Strengths and weaknesses]
• Branch C Analysis: [Strengths and weaknesses]

OPTIMAL PATH SELECTION:
Selected Branch: [Choose the best branch with clear justification]
Justification: [Explain why this path is optimal]

FINAL SOLUTION:
[Complete solution using the optimal path with full reasoning]
        """.strip()
        
        return prompt
    
    def _get_approach_description(self, branch_number: int) -> str:
        """Get description for different approach types."""
        approaches = {
            1: "Direct/Analytical Approach",
            2: "Alternative/Creative Approach", 
            3: "Systematic/Methodical Approach",
            4: "Hybrid/Combined Approach"
        }
        return approaches.get(branch_number, f"Approach {branch_number}")
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse ToT response to extract branches and evaluations."""
        parsed = {
            'branches': [],
            'evaluations': {},
            'selected_branch': '',
            'final_solution': '',
            'comparison_performed': False
        }
        
        # Extract branches
        branch_pattern = r'Branch ([A-Z]):\s*(.+?)(?=Branch [A-Z]:|PATH COMPARISON|$)'
        branches = re.findall(branch_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for branch_id, branch_content in branches:
            # Extract steps for this branch
            step_pattern = f'Step {branch_id}\\d+:(.+?)(?=Step {branch_id}\\d+:|Evaluation:|Branch [A-Z]:|$)'
            steps = re.findall(step_pattern, branch_content, re.DOTALL)
            
            # Extract evaluation for this branch
            eval_pattern = f'Evaluation:(.+?)(?=Branch [A-Z]:|PATH COMPARISON|$)'
            eval_match = re.search(eval_pattern, branch_content, re.DOTALL | re.IGNORECASE)
            evaluation = eval_match.group(1).strip() if eval_match else ""
            
            parsed['branches'].append({
                'id': branch_id,
                'steps': [step.strip() for step in steps],
                'evaluation': evaluation
            })
        
        # Extract selected branch
        selected_pattern = r'Selected Branch:\s*(.+?)(?=Justification:|FINAL SOLUTION:|$)'
        selected_match = re.search(selected_pattern, response, re.IGNORECASE)
        if selected_match:
            parsed['selected_branch'] = selected_match.group(1).strip()
        
        # Extract final solution
        solution_pattern = r'FINAL SOLUTION:\s*(.+?)$'
        solution_match = re.search(solution_pattern, response, re.DOTALL | re.IGNORECASE)
        if solution_match:
            parsed['final_solution'] = solution_match.group(1).strip()
        
        # Check if comparison was performed
        parsed['comparison_performed'] = 'PATH COMPARISON' in response.upper()
        
        return parsed

class ReActFramework(BasePromptFramework):
    """
    Implementation of ReAct (Reasoning + Acting) framework.
    
    Based on Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models"
    """
    
    def __init__(self):
        super().__init__(
            name="ReAct",
            description="Reasoning and Acting integration framework"
        )
        self.available_actions = [
            'search', 'calculate', 'analyze', 'verify', 'synthesize'
        ]
    
    def generate_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate a ReAct prompt."""
        context = context or {}
        available_tools = context.get('tools', self.available_actions)
        max_iterations = context.get('max_iterations', 5)
        
        tools_description = self._format_tools_description(available_tools)
        
        prompt = f"""
Task: {query}

Available Actions: {', '.join(available_tools)}

{tools_description}

I'll solve this using Reasoning + Acting cycles:

Thought 1: [Initial analysis of what I need to do]
Action 1: [Specific action to take - must be one of: {', '.join(available_tools)}]
Observation 1: [Result of the action or information gathered]

Thought 2: [Reasoning based on the observation]
Action 2: [Next action based on my reasoning]
Observation 2: [Result of the second action]

Thought 3: [Updated analysis incorporating all information]
Action 3: [Final action or conclusion]
Observation 3: [Final result or synthesis]

Final Answer: [Complete solution based on the reasoning-action cycle]

Note: Continue the Thought-Action-Observation cycle until you have sufficient information to provide a comprehensive answer.
        """.strip()
        
        return prompt
    
    def _format_tools_description(self, tools: List[str]) -> str:
        """Format description of available tools."""
        tool_descriptions = {
            'search': "Search for relevant information",
            'calculate': "Perform mathematical calculations", 
            'analyze': "Analyze data or information",
            'verify': "Verify facts or check accuracy",
            'synthesize': "Combine information from multiple sources"
        }
        
        descriptions = []
        for tool in tools:
            desc = tool_descriptions.get(tool, f"Use {tool} functionality")
            descriptions.append(f"- {tool}: {desc}")
        
        return "Tool Descriptions:\n" + "\n".join(descriptions)
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse ReAct response to extract thought-action cycles."""
        parsed = {
            'cycles': [],
            'final_answer': '',
            'total_cycles': 0,
            'actions_used': []
        }
        
        # Extract thought-action-observation cycles
        cycle_pattern = r'Thought (\d+):\s*(.+?)Action \1:\s*(.+?)Observation \1:\s*(.+?)(?=Thought \d+:|Final Answer:|$)'
        cycles = re.findall(cycle_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for cycle_num, thought, action, observation in cycles:
            cycle_data = {
                'cycle_number': int(cycle_num),
                'thought': thought.strip(),
                'action': action.strip(),
                'observation': observation.strip()
            }
            parsed['cycles'].append(cycle_data)
            
            # Track actions used
            action_clean = action.strip().lower()
            for available_action in self.available_actions:
                if available_action in action_clean:
                    parsed['actions_used'].append(available_action)
                    break
        
        parsed['total_cycles'] = len(cycles)
        
        # Extract final answer
        final_pattern = r'Final Answer:\s*(.+?)$'
        final_match = re.search(final_pattern, response, re.DOTALL | re.IGNORECASE)
        if final_match:
            parsed['final_answer'] = final_match.group(1).strip()
        
        return parsed

class SelfConsistencyFramework(BasePromptFramework):
    """
    Implementation of Self-Consistency prompting framework.
    
    Based on Wang et al. (2022) "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
    """
    
    def __init__(self):
        super().__init__(
            name="Self-Consistency",
            description="Multiple reasoning paths with consistency validation"
        )
        self.num_paths = 3
        self.confidence_threshold = 0.7
    
    def generate_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate a Self-Consistency prompt."""
        context = context or {}
        num_paths = context.get('num_paths', self.num_paths)
        
        prompt = f"""
Problem: {query}

I'll solve this problem multiple ways to ensure consistency and reliability:

🔄 SOLUTION PATH 1:
[Solve the problem using your first approach - show all reasoning steps]

Answer 1: [State your answer clearly]

🔄 SOLUTION PATH 2:
[Solve the same problem using a different approach or perspective]

Answer 2: [State your answer clearly]

🔄 SOLUTION PATH 3:
[Solve the problem using a third approach for verification]

Answer 3: [State your answer clearly]

CONSISTENCY ANALYSIS:
• Agreement Check: [Do the answers agree? Which ones match?]
• Disagreement Analysis: [If answers differ, why might that be?]
• Reasoning Quality: [Which reasoning path seems most robust?]
• Confidence Assessment: [How confident are you in each approach?]

FINAL VALIDATED ANSWER:
Answer: [The most consistent and well-reasoned result]
Confidence Level: [High/Medium/Low with justification]
Reasoning: [Explanation of why this answer is most reliable]
        """.strip()
        
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse Self-Consistency response to extract multiple solutions."""
        parsed = {
            'solution_paths': [],
            'answers': [],
            'consistency_analysis': '',
            'final_answer': '',
            'confidence_level': '',
            'agreement_score': 0.0
        }
        
        # Extract solution paths and answers
        path_pattern = r'SOLUTION PATH (\d+):\s*(.+?)Answer \1:\s*(.+?)(?=SOLUTION PATH \d+:|CONSISTENCY ANALYSIS:|$)'
        paths = re.findall(path_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for path_num, reasoning, answer in paths:
            parsed['solution_paths'].append({
                'path_number': int(path_num),
                'reasoning': reasoning.strip(),
                'answer': answer.strip()
            })
            parsed['answers'].append(answer.strip())
        
        # Extract consistency analysis
        analysis_pattern = r'CONSISTENCY ANALYSIS:\s*(.+?)(?=FINAL VALIDATED ANSWER:|$)'
        analysis_match = re.search(analysis_pattern, response, re.DOTALL | re.IGNORECASE)
        if analysis_match:
            parsed['consistency_analysis'] = analysis_match.group(1).strip()
        
        # Extract final answer and confidence
        final_pattern = r'Answer:\s*(.+?)Confidence Level:\s*(.+?)(?=Reasoning:|$)'
        final_match = re.search(final_pattern, response, re.DOTALL | re.IGNORECASE)
        if final_match:
            parsed['final_answer'] = final_match.group(1).strip()
            parsed['confidence_level'] = final_match.group(2).strip()
        
        # Calculate agreement score
        if len(parsed['answers']) > 1:
            unique_answers = len(set(parsed['answers']))
            total_answers = len(parsed['answers'])
            parsed['agreement_score'] = 1.0 - (unique_answers - 1) / max(1, total_answers - 1)
        
        return parsed

class RoleBasedPromptingFramework(BasePromptFramework):
    """
    Implementation of Role-Based prompting framework.
    
    Leverages persona instantiation for specialized expertise access.
    """
    
    def __init__(self):
        super().__init__(
            name="Role-Based Prompting",
            description="Expert persona instantiation framework"
        )
        self.expert_roles = {
            'scientist': {
                'expertise': ['research methodology', 'data analysis', 'hypothesis testing'],
                'approach': 'evidence-based, systematic, peer-review oriented',
                'communication': 'precise, technical, well-cited'
            },
            'analyst': {
                'expertise': ['pattern recognition', 'data interpretation', 'strategic thinking'],
                'approach': 'analytical, data-driven, objective',
                'communication': 'clear, structured, insight-focused'
            },
            'engineer': {
                'expertise': ['problem-solving', 'system design', 'optimization'],
                'approach': 'pragmatic, efficient, solution-oriented',
                'communication': 'practical, detailed, implementation-focused'
            },
            'teacher': {
                'expertise': ['explanation', 'curriculum design', 'learning psychology'],
                'approach': 'pedagogical, progressive, adaptive',
                'communication': 'clear, engaging, scaffolded'
            }
        }
    
    def generate_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate a role-based prompt."""
        context = context or {}
        role = context.get('role', 'analyst')
        experience_level = context.get('experience_level', 'senior')
        
        if role not in self.expert_roles:
            role = 'analyst'
        
        role_config = self.expert_roles[role]
        
        prompt = f"""
EXPERT ROLE ACTIVATION:

You are a {experience_level} {role} with extensive professional experience.

Your Expertise Includes:
{self._format_expertise(role_config['expertise'])}

Your Professional Approach:
{role_config['approach']}

Your Communication Style:
{role_config['communication']}

Professional Methodology:
When facing complex problems, you:
1. Apply domain-specific frameworks and best practices
2. Draw upon extensive professional experience
3. Consider multiple perspectives and potential solutions
4. Validate approaches against industry standards
5. Communicate findings clearly and actionably

TASK: {query}

Please approach this task with your full professional expertise and experience. Structure your response in a way that reflects your professional standards and methodology.

PROFESSIONAL RESPONSE:
        """.strip()
        
        return prompt
    
    def _format_expertise(self, expertise_list: List[str]) -> str:
        """Format expertise list for prompt."""
        return '\n'.join(f"• {expertise}" for expertise in expertise_list)
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse role-based response to extract professional elements."""
        parsed = {
            'professional_structure': False,
            'domain_expertise_evident': False,
            'methodology_applied': False,
            'communication_quality': 'unknown',
            'expertise_indicators': []
        }
        
        # Check for professional structure
        structure_indicators = ['analysis:', 'methodology:', 'recommendation:', 'conclusion:']
        parsed['professional_structure'] = any(
            indicator in response.lower() for indicator in structure_indicators
        )
        
        # Check for domain expertise indicators
        expertise_indicators = ['based on experience', 'industry standard', 'best practice', 
                              'professional judgment', 'methodology', 'framework']
        found_indicators = [ind for ind in expertise_indicators if ind in response.lower()]
        parsed['expertise_indicators'] = found_indicators
        parsed['domain_expertise_evident'] = len(found_indicators) > 0
        
        # Check for methodology application
        methodology_indicators = ['step 1', 'first', 'analysis', 'evaluation', 'assessment']
        parsed['methodology_applied'] = any(
            indicator in response.lower() for indicator in methodology_indicators
        )
        
        return parsed

class MetaCognitivePromptingFramework(BasePromptFramework):
    """
    Implementation of Meta-Cognitive prompting framework.
    
    Enables self-reflection, error correction, and iterative improvement.
    """
    
    def __init__(self):
        super().__init__(
            name="Meta-Cognitive Prompting",
            description="Self-reflective reasoning and improvement framework"
        )
        self.reflection_aspects = [
            'completeness', 'accuracy', 'clarity', 'relevance', 'methodology'
        ]
    
    def generate_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate a meta-cognitive prompt."""
        context = context or {}
        reflection_depth = context.get('reflection_depth', 'standard')
        
        if reflection_depth == 'deep':
            return self._generate_deep_metacognitive_prompt(query)
        else:
            return self._generate_standard_metacognitive_prompt(query)
    
    def _generate_standard_metacognitive_prompt(self, query: str) -> str:
        """Generate standard meta-cognitive prompt."""
        return f"""
Task: {query}

I'll approach this with meta-cognitive awareness, monitoring and improving my reasoning:

INITIAL RESPONSE:
[Provide your initial response to the task]

META-COGNITIVE REFLECTION:
Now let me reflect on my response:

Quality Assessment:
• Completeness: Did I address all aspects of the task?
• Accuracy: Are my facts and reasoning correct? 
• Clarity: Is my explanation clear and well-structured?
• Relevance: Does everything relate directly to the task?
• Methodology: Did I use appropriate reasoning methods?

Self-Critique:
• What might I have missed or overlooked?
• Are there alternative perspectives I should consider?
• Could I provide better examples or explanations?
• Are there any weaknesses in my reasoning?

IMPROVED RESPONSE:
Based on my reflection, here's my enhanced response:
[Provide an improved version that addresses identified gaps]

FINAL REFLECTION:
[Brief assessment of the improvement process and remaining limitations]
        """.strip()
    
    def _generate_deep_metacognitive_prompt(self, query: str) -> str:
        """Generate deep meta-cognitive prompt with multiple reflection layers."""
        return f"""
Task: {query}

I'll use deep meta-cognitive reflection with multiple improvement cycles:

CYCLE 1 - INITIAL RESPONSE:
[Provide initial response]

CYCLE 1 - META-ANALYSIS:
• Cognitive processes used: [What thinking strategies did I employ?]
• Assumptions made: [What did I assume without verification?]
• Knowledge gaps: [What information do I lack?]
• Reasoning quality: [How sound is my logic?]

CYCLE 2 - REFINED RESPONSE:
[Improved response addressing cycle 1 analysis]

CYCLE 2 - DEEPER REFLECTION:
• Alternative approaches: [What other methods could I use?]
• Perspective diversity: [Have I considered multiple viewpoints?]
• Error potential: [Where might I be wrong?]
• Confidence calibration: [How confident should I be?]

CYCLE 3 - OPTIMIZED RESPONSE:
[Final optimized response incorporating all reflections]

META-LEARNING:
• What did I learn about my thinking process?
• How can I improve my approach to similar problems?
• What patterns do I notice in my reasoning?
        """.strip()
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse meta-cognitive response to extract reflection elements."""
        parsed = {
            'initial_response': '',
            'reflections': [],
            'improvements': [],
            'meta_learning': '',
            'reflection_depth': 0,
            'self_awareness_indicators': []
        }
        
        # Extract initial response
        initial_pattern = r'INITIAL RESPONSE:\s*(.+?)(?=META-COGNITIVE REFLECTION:|CYCLE 1|$)'
        initial_match = re.search(initial_pattern, response, re.DOTALL | re.IGNORECASE)
        if initial_match:
            parsed['initial_response'] = initial_match.group(1).strip()
        
        # Extract reflection cycles
        reflection_pattern = r'(?:META-COGNITIVE REFLECTION:|CYCLE \d+ - META-ANALYSIS:|CYCLE \d+ - DEEPER REFLECTION:)\s*(.+?)(?=IMPROVED RESPONSE:|CYCLE \d+|FINAL REFLECTION:|META-LEARNING:|$)'
        reflections = re.findall(reflection_pattern, response, re.DOTALL | re.IGNORECASE)
        parsed['reflections'] = [ref.strip() for ref in reflections]
        parsed['reflection_depth'] = len(reflections)
        
        # Extract improvements
        improvement_pattern = r'(?:IMPROVED RESPONSE:|CYCLE \d+ - REFINED RESPONSE:|CYCLE \d+ - OPTIMIZED RESPONSE:)\s*(.+?)(?=FINAL REFLECTION:|META-COGNITIVE REFLECTION:|CYCLE \d+|META-LEARNING:|$)'
        improvements = re.findall(improvement_pattern, response, re.DOTALL | re.IGNORECASE)
        parsed['improvements'] = [imp.strip() for imp in improvements]
        
        # Extract meta-learning
        learning_pattern = r'(?:FINAL REFLECTION:|META-LEARNING:)\s*(.+?)$'
        learning_match =
