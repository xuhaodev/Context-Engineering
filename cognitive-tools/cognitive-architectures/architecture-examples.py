"""
Cognitive Architecture Examples - Practical implementations of Solver, Tutor, and Research architectures.

This module demonstrates how the theoretical frameworks presented in the cognitive architecture 
documentation can be practically implemented and applied to real-world problems.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import random
from datetime import datetime
import math
import re
from collections import defaultdict

# =============================================================================
# CORE UTILITIES AND SHARED COMPONENTS
# =============================================================================

def generate_id() -> str:
    """Generate a unique identifier."""
    return f"id_{random.randint(10000, 99999)}_{int(datetime.now().timestamp())}"

def get_current_timestamp() -> str:
    """Get the current timestamp as a string."""
    return datetime.now().isoformat()

# Mock LLM executor for demonstration
def llm_executor(prompt: str) -> str:
    """
    Simulates execution of prompts through an LLM.
    
    In a real implementation, this would connect to an actual LLM API.
    """
    print(f"\n[LLM EXECUTOR] Processing prompt: {prompt[:100]}...")
    
    # Simulate different responses based on prompt content
    if "understand" in prompt.lower():
        return """{"understanding": {
            "problem_type": "algebraic equation",
            "variables": ["x"],
            "constraints": ["x must be a real number"],
            "goal": "find the value of x that satisfies the equation"
        }}"""
    elif "analyze" in prompt.lower():
        return """{"analysis": {
            "approach": "solve for x by isolating it on one side",
            "steps": ["combine like terms", "divide both sides by coefficient"],
            "expected_complexity": "low"
        }}"""
    elif "synthesize" in prompt.lower():
        return """{"synthesis": {
            "key_findings": ["concept A relates to concept B", "evidence supports hypothesis H1"],
            "patterns": ["temporal trend increasing", "correlation between X and Y"],
            "contradictions": ["study 1 and study 2 have conflicting results"],
            "gaps": ["no research on factor Z"]
        }}"""
    elif "hypothesis" in prompt.lower():
        return """{"hypothesis": {
            "statement": "Increased exposure to X leads to improved Y under conditions Z",
            "variables": {"independent": "X exposure", "dependent": "Y performance", "moderator": "Z conditions"},
            "testability": "high",
            "theoretical_grounding": "consistent with theory T"
        }}"""
    elif "explain" in prompt.lower():
        return """{"explanation": {
            "concept": "mathematical concept clearly explained",
            "examples": ["example 1", "example 2"],
            "analogies": ["real-world analogy that clarifies concept"],
            "potential_misconceptions": ["common misconception addressed"]
        }}"""
    
    # Generic fallback response
    return f"Simulated LLM response for: {prompt[:50]}..."

def execute_protocol(protocol: str) -> Dict[str, Any]:
    """
    Execute a protocol shell and parse the result.
    
    Args:
        protocol: Protocol shell to execute
        
    Returns:
        dict: Parsed protocol results
    """
    # Execute through LLM
    response = llm_executor(protocol)
    
    # Try to parse as JSON
    try:
        if isinstance(response, str):
            # Check if response looks like JSON
            if response.strip().startswith('{') and response.strip().endswith('}'):
                return json.loads(response)
        
        # If already a dict or parsing failed, return as is
        if isinstance(response, dict):
            return response
        
        # Create a simple wrapper if not parseable
        return {"raw_response": response}
        
    except Exception as e:
        print(f"[ERROR] Failed to parse protocol response: {e}")
        return {"error": str(e), "raw_response": response}

# =============================================================================
# PROTOCOL SHELL IMPLEMENTATION
# =============================================================================

class ProtocolShell:
    """Implementation of the protocol shell framework."""
    
    def __init__(self, intent: str, input_params: Dict[str, Any], 
                 process_steps: List[Dict[str, str]], output_spec: Dict[str, str]):
        """
        Initialize a protocol shell.
        
        Args:
            intent: Clear statement of purpose
            input_params: Input parameters
            process_steps: Ordered process steps
            output_spec: Expected output specification
        """
        self.intent = intent
        self.input_params = input_params
        self.process_steps = process_steps
        self.output_spec = output_spec
        self.execution_trace = []
    
    def to_prompt(self) -> str:
        """Convert protocol shell to structured prompt format."""
        # Generate a protocol name based on intent if not explicitly provided
        protocol_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.intent.lower().replace(' ', '_'))
        
        # Format input parameters
        input_params_str = ",\n        ".join([f"{k}={self._format_value(v)}" 
                                             for k, v in self.input_params.items()])
        
        # Format process steps
        process_steps_str = ",\n        ".join([f"/{step['action']}{{action=\"{step['description']}\"" +
                                              (f", tools={self._format_value(step.get('tools', []))}" 
                                               if 'tools' in step else "") + "}"
                                              for step in self.process_steps])
        
        # Format output specification
        output_spec_str = ",\n        ".join([f"{k}=\"{v}\"" 
                                            for k, v in self.output_spec.items()])
        
        # Construct the complete protocol prompt
        prompt = f"""
        /{protocol_name}{{
            intent="{self.intent}",
            input={{
                {input_params_str}
            }},
            process=[
                {process_steps_str}
            ],
            output={{
                {output_spec_str}
            }}
        }}
        """
        
        return prompt
    
    def _format_value(self, v: Any) -> str:
        """Format values appropriately based on type."""
        if isinstance(v, str):
            return f'"{v}"'
        elif isinstance(v, (list, tuple)):
            items = [self._format_value(item) for item in v]
            return f"[{', '.join(items)}]"
        elif isinstance(v, dict):
            items = [f"{k}: {self._format_value(v)}" for k, v in v.items()]
            return f"{{{', '.join(items)}}}"
        else:
            return str(v)
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the protocol shell.
        
        Returns:
            dict: Results of protocol execution
        """
        prompt = self.to_prompt()
        
        # Execute the protocol through LLM
        result = execute_protocol(prompt)
        
        # Record execution trace
        self.execution_trace.append({
            "timestamp": get_current_timestamp(),
            "prompt": prompt,
            "result": result
        })
        
        return result

# =============================================================================
# SEMANTIC FIELD IMPLEMENTATION
# =============================================================================

class SemanticField:
    """Base implementation of semantic field concepts for all architectures."""
    
    def __init__(self, dimensions: int = 128, name: str = "generic_field"):
        """
        Initialize a semantic field.
        
        Args:
            dimensions: Dimensionality of the field
            name: Name of the field
        """
        self.dimensions = dimensions
        self.name = name
        self.field_state = np.zeros((dimensions,))
        self.attractors = {}
        self.boundaries = {}
        self.trajectories = []
        self.residue = []
    
    def add_attractor(self, concept: str, position: np.ndarray = None, 
                     strength: float = 1.0, basin_shape: str = "gaussian") -> Dict[str, Any]:
        """
        Add an attractor to the field.
        
        Args:
            concept: Concept associated with the attractor
            position: Position in field space (random if None)
            strength: Attractor strength
            basin_shape: Shape of attractor basin
            
        Returns:
            dict: Attractor information
        """
        # Generate position if not provided
        if position is None:
            position = np.random.normal(0, 1, self.dimensions)
            position = position / np.linalg.norm(position)
        
        # Ensure position has correct dimensions
        if len(position) != self.dimensions:
            position = np.resize(position, (self.dimensions,))
        
        # Generate ID for attractor
        attractor_id = f"attr_{concept.replace(' ', '_')}_{generate_id()}"
        
        # Create attractor
        self.attractors[attractor_id] = {
            "concept": concept,
            "position": position,
            "strength": strength,
            "basin_shape": basin_shape,
            "created_at": get_current_timestamp()
        }
        
        # Update field state based on new attractor
        self._update_field_state()
        
        return self.attractors[attractor_id]
    
    def _update_field_state(self):
        """Update the field state based on attractors and boundaries."""
        # Start with zero field
        new_state = np.zeros((self.dimensions,))
        
        # Add influence of each attractor
        for attractor_id, attractor in self.attractors.items():
            position = attractor["position"]
            strength = attractor["strength"]
            basin_shape = attractor["basin_shape"]
            
            # Different basin shapes have different influence patterns
            if basin_shape == "gaussian":
                # Gaussian influence that falls off with distance
                for i in range(self.dimensions):
                    # Simplified: just add weighted position
                    new_state[i] += position[i] * strength
            
            # Other basin shapes could be implemented similarly
        
        # Normalize the field state
        if np.linalg.norm(new_state) > 0:
            new_state = new_state / np.linalg.norm(new_state)
        
        # Store the updated state
        self.field_state = new_state
    
    def calculate_trajectory(self, start_state: np.ndarray, steps: int = 10) -> List[np.ndarray]:
        """
        Calculate a trajectory through the field from a starting state.
        
        Args:
            start_state: Starting position in field space
            steps: Number of steps to simulate
            
        Returns:
            list: Sequence of states forming a trajectory
        """
        trajectory = [start_state]
        current_state = start_state.copy()
        
        for _ in range(steps):
            # Calculate the influence of all attractors
            next_state = current_state.copy()
            
            for attractor_id, attractor in self.attractors.items():
                position = attractor["position"]
                strength = attractor["strength"]
                
                # Vector from current state to attractor
                direction = position - current_state
                
                # Normalize
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                
                # Move towards attractor based on strength and distance
                # Simplified model: attraction decreases with square of distance
                distance = np.linalg.norm(position - current_state)
                if distance > 0:
                    attraction = strength / (distance * distance)
                    next_state += direction * attraction
            
            # Normalize the next state
            if np.linalg.norm(next_state) > 0:
                next_state = next_state / np.linalg.norm(next_state)
            
            # Add to trajectory and update current state
            trajectory.append(next_state)
            current_state = next_state
        
        # Record the trajectory
        self.trajectories.append({
            "start": start_state,
            "steps": trajectory,
            "created_at": get_current_timestamp()
        })
        
        return trajectory
    
    def detect_basins(self) -> List[Dict[str, Any]]:
        """
        Detect basin regions in the field.
        
        Returns:
            list: Detected basin regions
        """
        basins = []
        
        # For each attractor, identify its basin of attraction
        for attractor_id, attractor in self.attractors.items():
            # Basin properties would depend on attractor and field state
            basin = {
                "attractor_id": attractor_id,
                "concept": attractor["concept"],
                "center": attractor["position"],
                "radius": 0.2 + 0.3 * attractor["strength"],  # Simplified radius calculation
                "strength": attractor["strength"]
            }
            
            basins.append(basin)
        
        return basins
    
    def visualize(self, show_attractors: bool = True, show_trajectories: bool = True, 
                 reduced_dims: int = 2) -> plt.Figure:
        """
        Visualize the field in reduced dimensions.
        
        Args:
            show_attractors: Whether to show attractors
            show_trajectories: Whether to show trajectories
            reduced_dims: Dimensionality for visualization
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # For visualization, we'll reduce to 2D using a simple approach
        # In a real implementation, PCA or t-SNE would be more appropriate
        
        # Function to reduce dimensions
        def reduce_dims(vector):
            if reduced_dims == 2:
                return vector[:2] if len(vector) >= 2 else np.pad(vector, (0, 2 - len(vector)))
            else:
                return vector[:reduced_dims] if len(vector) >= reduced_dims else np.pad(vector, (0, reduced_dims - len(vector)))
        
        # Plot field boundaries (simplified as a circle for 2D)
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        ax.add_artist(circle)
        
        # Plot attractors
        if show_attractors and self.attractors:
            for attractor_id, attractor in self.attractors.items():
                pos = reduce_dims(attractor["position"])
                strength = attractor["strength"]
                
                # Plot attractor point
                ax.scatter(pos[0], pos[1], s=100 * strength, color='red', alpha=0.7)
                
                # Plot attractor label
                ax.text(pos[0], pos[1], attractor["concept"], fontsize=9, ha='center')
                
                # Plot basin of attraction (simplified as a circle)
                basin_circle = plt.Circle((pos[0], pos[1]), 0.2 * strength, fill=True, 
                                        color='red', alpha=0.1)
                ax.add_artist(basin_circle)
        
        # Plot trajectories
        if show_trajectories and self.trajectories:
            for trajectory in self.trajectories:
                points = [reduce_dims(step) for step in trajectory["steps"]]
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                
                # Plot trajectory line
                ax.plot(x_vals, y_vals, 'b-', alpha=0.5)
                
                # Plot start and end points
                ax.scatter(x_vals[0], y_vals[0], color='green', s=50, label='Start')
                ax.scatter(x_vals[-1], y_vals[-1], color='blue', s=50, label='End')
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # Set title and labels
        ax.set_title(f"Semantic Field: {self.name}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        return fig

# =============================================================================
# SOLVER ARCHITECTURE IMPLEMENTATION
# =============================================================================

class CognitiveToolsLibrary:
    """Implementation of cognitive tools for problem-solving."""
    
    @staticmethod
    def understand_question(question: str, domain: str = None) -> Dict[str, Any]:
        """
        Break down and comprehend a problem statement.
        
        Args:
            question: The problem to be understood
            domain: Optional domain context
            
        Returns:
            dict: Structured problem understanding
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Break down and comprehend the problem thoroughly",
            input_params={
                "question": question,
                "domain": domain if domain else "general"
            },
            process_steps=[
                {"action": "extract", "description": "Identify key components of the problem"},
                {"action": "identify", "description": "Detect variables, constants, and unknowns"},
                {"action": "determine", "description": "Identify goals and objectives"},
                {"action": "recognize", "description": "Identify constraints and conditions"},
                {"action": "classify", "description": "Classify problem type and domain"}
            ],
            output_spec={
                "components": "Identified key elements",
                "variables": "Detected variables and unknowns",
                "goals": "Primary objectives to achieve",
                "constraints": "Limitations and conditions",
                "problem_type": "Classification of problem"
            }
        )
        
        # Execute protocol
        return protocol.execute()
    
    @staticmethod
    def decompose_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompose a complex problem into simpler subproblems.
        
        Args:
            problem: Structured problem representation
            
        Returns:
            dict: Decomposed problem structure
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Decompose complex problem into manageable subproblems",
            input_params={"problem": problem},
            process_steps=[
                {"action": "analyze", "description": "Analyze problem structure"},
                {"action": "identify", "description": "Identify natural subproblems"},
                {"action": "organize", "description": "Determine subproblem dependencies"},
                {"action": "simplify", "description": "Reduce complexity of each subproblem"}
            ],
            output_spec={
                "subproblems": "List of identified subproblems",
                "dependencies": "Relationships between subproblems",
                "sequence": "Recommended solution sequence",
                "simplification": "How each subproblem is simplified"
            }
        )
        
        # Execute protocol
        return protocol.execute()
    
    @staticmethod
    def step_by_step(problem: Dict[str, Any], approach: str) -> Dict[str, Any]:
        """
        Generate a step-by-step solution plan.
        
        Args:
            problem: Structured problem representation
            approach: Solution approach to use
            
        Returns:
            dict: Step-by-step solution
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Generate detailed step-by-step solution",
            input_params={
                "problem": problem,
                "approach": approach
            },
            process_steps=[
                {"action": "plan", "description": "Plan solution steps"},
                {"action": "execute", "description": "Execute each step in sequence"},
                {"action": "track", "description": "Track progress and intermediate results"},
                {"action": "verify", "description": "Verify each step's correctness"}
            ],
            output_spec={
                "steps": "Ordered solution steps",
                "explanation": "Explanation for each step",
                "intermediate_results": "Results after each step",
                "final_solution": "Complete solution"
            }
        )
        
        # Execute protocol
        return protocol.execute()
    
    @staticmethod
    def verify_solution(problem: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify the correctness of a solution.
        
        Args:
            problem: Structured problem representation
            solution: Proposed solution
            
        Returns:
            dict: Verification results
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Verify solution correctness and completeness",
            input_params={
                "problem": problem,
                "solution": solution
            },
            process_steps=[
                {"action": "check", "description": "Check solution against problem constraints"},
                {"action": "test", "description": "Test solution with examples or edge cases"},
                {"action": "analyze", "description": "Analyze for errors or inefficiencies"},
                {"action": "evaluate", "description": "Evaluate overall solution quality"}
            ],
            output_spec={
                "is_correct": "Whether the solution is correct",
                "verification_details": "Details of verification process",
                "errors": "Any identified errors",
                "improvements": "Potential improvements",
                "confidence": "Confidence in solution correctness"
            }
        )
        
        # Execute protocol
        return protocol.execute()

class MetaCognitiveController:
    """Implementation of metacognitive monitoring and regulation."""
    
    def __init__(self):
        """Initialize the metacognitive controller."""
        self.state = {
            "current_stage": None,
            "progress": {},
            "obstacles": [],
            "strategy_adjustments": [],
            "insights": []
        }
    
    def monitor(self, phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor progress and detect obstacles.
        
        Args:
            phase_results: Results from current problem-solving phase
            
        Returns:
            dict: Monitoring assessment
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Track progress and identify obstacles",
            input_params={
                "phase": self.state["current_stage"],
                "results": phase_results
            },
            process_steps=[
                {"action": "assess", "description": "Evaluate progress against expected outcomes"},
                {"action": "detect", "description": "Identify obstacles, challenges, or limitations"},
                {"action": "identify", "description": "Identify uncertainty or knowledge gaps"},
                {"action": "measure", "description": "Measure confidence in current approach"}
            ],
            output_spec={
                "progress_assessment": "Evaluation of current progress",
                "obstacles": "Identified challenges or blockers",
                "uncertainty": "Areas of limited confidence",
                "recommendations": "Suggested adjustments"
            }
        )
        
        # Execute protocol
        monitoring_results = protocol.execute()
        
        # Update state with monitoring results
        self.state["progress"][self.state["current_stage"]] = monitoring_results["progress_assessment"]
        
        if "obstacles" in monitoring_results and isinstance(monitoring_results["obstacles"], list):
            self.state["obstacles"].extend(monitoring_results["obstacles"])
        
        return monitoring_results
    
    def regulate(self, monitoring_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust strategy based on monitoring.
        
        Args:
            monitoring_assessment: Results from monitoring
            
        Returns:
            dict: Strategy adjustments
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Adjust strategy to overcome obstacles",
            input_params={
                "current_phase": self.state["current_stage"],
                "assessment": monitoring_assessment,
                "history": self.state
            },
            process_steps=[
                {"action": "evaluate", "description": "Evaluate current strategy effectiveness"},
                {"action": "generate", "description": "Generate alternative approaches"},
                {"action": "select", "description": "Select most promising adjustments"},
                {"action": "formulate", "description": "Formulate implementation plan"}
            ],
            output_spec={
                "strategy_assessment": "Evaluation of current strategy",
                "adjustments": "Recommended strategy changes",
                "implementation": "How to apply adjustments",
                "expected_outcomes": "Anticipated improvements"
            }
        )
        
        # Execute protocol
        regulation_results = protocol.execute()
        
        # Update state with regulation results
        if "adjustments" in regulation_results:
            self.state["strategy_adjustments"].append(regulation_results["adjustments"])
        
        return regulation_results
    
    def reflect(self, complete_process: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on the entire problem-solving process.
        
        Args:
            complete_process: The full problem-solving trace
            
        Returns:
            dict: Reflection insights and learning
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Extract insights and improve future problem-solving",
            input_params={
                "complete_process": complete_process
            },
            process_steps=[
                {"action": "analyze", "description": "Analyze effectiveness of overall approach"},
                {"action": "identify", "description": "Identify strengths and weaknesses"},
                {"action": "extract", "description": "Extract generalizable patterns and insights"},
                {"action": "formulate", "description": "Formulate lessons for future problems"}
            ],
            output_spec={
                "effectiveness": "Assessment of problem-solving approach",
                "strengths": "What worked particularly well",
                "weaknesses": "Areas for improvement",
                "patterns": "Identified recurring patterns",
                "insights": "Key learnings",
                "future_recommendations": "How to improve future problem-solving"
            }
        )
        
        # Execute protocol
        reflection_results = protocol.execute()
        
        # Update state with reflection results
        if "insights" in reflection_results:
            self.state["insights"] = reflection_results["insights"]
        
        return reflection_results

class SolverArchitecture:
    """Complete implementation of the Solver Architecture."""
    
    def __init__(self):
        """Initialize the solver architecture."""
        self.tools_library = CognitiveToolsLibrary()
        self.metacognitive_controller = MetaCognitiveController()
        self.field = SemanticField(name="solution_field")
        self.session_history = []
    
    def solve(self, problem: str, domain: str = None) -> Dict[str, Any]:
        """
        Solve a problem using the complete architecture.
        
        Args:
            problem: Problem statement
            domain: Optional domain context
            
        Returns:
            dict: Solution and reasoning trace
        """
        # Initialize session
        session = {
            "problem": problem,
            "domain": domain,
            "stages": {},
            "solution": None,
            "meta": {},
            "field_state": {}
        }
        
        # 1. UNDERSTAND stage
        self.metacognitive_controller.state["current_stage"] = "understand"
        understanding = self.tools_library.understand_question(problem, domain)
        session["stages"]["understand"] = understanding
        
        # Monitor understanding progress
        understanding_assessment = self.metacognitive_controller.monitor(understanding)
        
        # If obstacles detected, adjust strategy
        if understanding_assessment.get("obstacles"):
            understanding_adjustment = self.metacognitive_controller.regulate(understanding_assessment)
            # In a real implementation, would apply adjustments to understanding
        
        # 2. ANALYZE stage
        self.metacognitive_controller.state["current_stage"] = "analyze"
        analysis = self.tools_library.decompose_problem(understanding)
        session["stages"]["analyze"] = analysis
        
        # Monitor analysis progress
        analysis_assessment = self.metacognitive_controller.monitor(analysis)
        
        # If obstacles detected, adjust strategy
        if analysis_assessment.get("obstacles"):
            analysis_adjustment = self.metacognitive_controller.regulate(analysis_assessment)
            # In a real implementation, would apply adjustments to analysis
        
        # Create solution approach
        approach = analysis.get("approach", "step_by_step")
        
        # 3. SOLVE stage
        self.metacognitive_controller.state["current_stage"] = "solve"
        solution = self.tools_library.step_by_step(understanding, approach)
        session["stages"]["solve"] = solution
        
        # Monitor solution progress
        solution_assessment = self.metacognitive_controller.monitor(solution)
        
        # If obstacles detected, adjust strategy
        if solution_assessment.get("obstacles"):
            solution_adjustment = self.metacognitive_controller.regulate(solution_assessment)
            # In a real implementation, would apply adjustments to solution
        
        # 4. VERIFY stage
        self.metacognitive_controller.state["current_stage"] = "verify"
        verification = self.tools_library.verify_solution(understanding, solution)
        session["stages"]["verify"] = verification
        
        # Monitor verification progress
        verification_assessment = self.metacognitive_controller.monitor(verification)
        
        # Final solution
        session["solution"] = solution.get("final_solution", "Solution not found")
        
        # Meta-cognitive reflection
        reflection = self.metacognitive_controller.reflect({
            "understanding": understanding,
            "analysis": analysis,
            "solution": solution,
            "verification": verification
        })
        
        session["meta"] = {
            "progress": self.metacognitive_controller.state["progress"],
            "obstacles": self.metacognitive_controller.state["obstacles"],
            "strategy_adjustments": self.metacognitive_controller.state["strategy_adjustments"],
            "insights": reflection.get("insights", [])
        }
        
        # Update field state
        self.update_field_from_solution(understanding, solution)
        session["field_state"] = {
            "attractors": len(self.field.attractors),
            "trajectories": len(self.field.trajectories)
        }
        
        # Add to session history
        self.session_history.append(session)
        
        return session
    
    def update_field_from_solution(self, understanding: Dict[str, Any], solution: Dict[str, Any]):
        """
        Update the semantic field based on the problem and solution.
        
        Args:
            understanding: Problem understanding
            solution: Problem solution
        """
        # Add problem as attractor
        problem_type = understanding.get("problem_type", "unknown")
        self.field.add_attractor(f"Problem: {problem_type}", 
                               np.random.normal(0, 1, self.field.dimensions),
                               strength=0.8)
        
        # Add solution approach as attractor
        solution_approach = solution.get("approach", "unknown")
        self.field.add_attractor(f"Approach: {solution_approach}",
                               np.random.normal(0, 1, self.field.dimensions),
                               strength=1.0)
        
        # Simulate a solution trajectory
        start_state = np.random.normal(0, 1, self.field.dimensions)
        start_state = start_state / np.linalg.norm(start_state)
        self.field.calculate_trajectory(start_state, steps=10)
    
    def visualize_solution_process(self, session_index: int = -1) -> plt.Figure:
        """
        Visualize the solution process from a session.
        
        Args:
            session_index: Index of session to visualize
            
        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        # Get the specified session
        
