#!/usr/bin/env python3
"""
Custom orchestrator for the Self-Optimizing API Router demo.

This orchestrator implements sophisticated quality evaluation and reward calculation
specifically designed for API routing scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import re
from typing import Dict, Any, List
from doa_framework import Orchestrator, SystemState, EpisodeTrajectory, RewardConfig


class APIRouterOrchestrator(Orchestrator):
    """
    Enhanced orchestrator for API routing scenarios.
    
    Features:
    - Task-specific quality evaluation
    - Efficiency vs quality trade-off optimization
    - Agent appropriateness scoring
    - Dynamic reward calculation
    """
    
    def run_episode(self, initial_state: SystemState) -> EpisodeTrajectory:
        """Run episode with enhanced quality evaluation."""
        trajectory = super().run_episode(initial_state)
        
        # Enhanced quality evaluation
        quality_score = self._evaluate_response_quality(
            initial_state.task_specification, 
            trajectory
        )
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency(trajectory)
        
        # Calculate agent appropriateness
        appropriateness_score = self._calculate_agent_appropriateness(
            initial_state.task_specification, 
            trajectory
        )
        
        # Update rewards based on comprehensive evaluation
        if trajectory.steps:
            # Calculate final reward components
            quality_reward = quality_score * 5.0  # Quality is important
            efficiency_reward = efficiency_score * 2.0  # Efficiency matters
            appropriateness_reward = appropriateness_score * 3.0  # Using right agents
            
            # Cost penalty
            total_cost = sum(self._estimate_step_cost(step) for step in trajectory.steps)
            cost_penalty = self.reward_config.lambda_cost_penalty * total_cost
            
            # Final reward calculation
            final_reward = quality_reward + efficiency_reward + appropriateness_reward - cost_penalty
            
            # Update final step reward
            final_step = trajectory.steps[-1]
            final_step.reward = final_reward
            
            # Update trajectory metrics
            trajectory.total_undiscounted_reward = sum(step.reward for step in trajectory.steps)
            # More lenient success criteria to encourage learning
            trajectory.task_successful = quality_score > 0.5 and efficiency_score > 0.25
        
        return trajectory
    
    def _evaluate_response_quality(self, task_specification: str, trajectory: EpisodeTrajectory) -> float:
        """
        Evaluate the quality of the final response based on task requirements.
        
        Args:
            task_specification: Original task specification (JSON string)
            trajectory: Complete episode trajectory
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            task_spec = json.loads(task_specification)
        except json.JSONDecodeError:
            task_spec = {"query": task_specification, "type": "unknown"}
        
        task_type = task_spec.get("type", "unknown")
        urgency = task_spec.get("urgency", "normal")
        detail_level = task_spec.get("detail_level", "summary")
        
        if not trajectory.steps:
            return 0.0
        
        # Analyze outputs from all agents in the trajectory
        quality_scores = []
        
        for step in trajectory.steps:
            agent = self.agents[step.agent_index]
            if agent.name == "TerminatorAgent":
                continue
            
            # Simulate agent execution for quality evaluation
            eval_state = SystemState(task_specification, [], 0, 5)
            output = agent.execute(eval_state)
            
            try:
                output_data = json.loads(str(output.content))
            except json.JSONDecodeError:
                output_data = {"content": str(output.content)}
            
            score = 0.0
            
            # Task-specific quality evaluation
            if task_type == "search":
                score = self._evaluate_search_quality(task_spec, output_data, agent.name)
            elif task_type == "recommendation":
                score = self._evaluate_recommendation_quality(task_spec, output_data, agent.name)
            elif task_type == "nlp_analysis":
                score = self._evaluate_nlp_quality(task_spec, output_data, agent.name)
            else:
                score = self._evaluate_generic_quality(task_spec, output_data, agent.name)
            
            quality_scores.append(score)
        
        # Return the maximum quality achieved (best agent output)
        return max(quality_scores) if quality_scores else 0.0
    
    def _evaluate_search_quality(self, task_spec: Dict, output_data: Dict, agent_name: str) -> float:
        """Evaluate quality for search tasks."""
        urgency = task_spec.get("urgency", "normal")
        query = task_spec.get("query", "")
        
        score = 0.0
        
        # Base quality by agent type
        if agent_name == "FastSearchAgent":
            score = 0.6  # Good for basic needs
            if urgency == "high":
                score += 0.2  # Bonus for matching urgency
        elif agent_name == "DeepAnalysisAgent":
            score = 0.9  # High quality comprehensive results
            if urgency == "low":
                score += 0.1  # Bonus when time allows
        
        # Content quality indicators
        if "results" in output_data:
            results = output_data.get("results", [])
            if len(results) >= 3:
                score += 0.1
        
        if "comprehensive" in str(output_data):
            score += 0.15
        
        if "confidence" in output_data:
            confidence = output_data.get("confidence", 0)
            if isinstance(confidence, (int, float)) and confidence > 0.8:
                score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_recommendation_quality(self, task_spec: Dict, output_data: Dict, agent_name: str) -> float:
        """Evaluate quality for recommendation tasks."""
        detail_level = task_spec.get("detail_level", "summary")
        
        score = 0.0
        
        # Base quality by agent type
        if agent_name == "FastSearchAgent":
            score = 0.5
            if detail_level == "summary":
                score += 0.2
        elif agent_name == "DeepAnalysisAgent":
            score = 0.8
            if detail_level == "full":
                score += 0.2
        
        # Content quality indicators
        if "recommendations" in output_data or "personalized_recommendations" in output_data:
            score += 0.1
        
        if "personalization" in str(output_data):
            score += 0.15
        
        if "confidence_score" in output_data:
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_nlp_quality(self, task_spec: Dict, output_data: Dict, agent_name: str) -> float:
        """Evaluate quality for NLP analysis tasks."""
        analysis_type = task_spec.get("analysis_type", "general")
        
        score = 0.0
        
        # NLP specialist should be heavily favored
        if agent_name == "NLPSpecialistAgent":
            score = 0.9
            if output_data.get("analysis_type") == analysis_type:
                score += 0.1  # Perfect match
        else:
            score = 0.2  # Other agents are not suitable
        
        # Content quality indicators
        if "sentiment" in output_data or "entities" in output_data:
            score += 0.1
        
        if "confidence" in output_data:
            score += 0.05
        
        return min(score, 1.0)
    
    def _evaluate_generic_quality(self, task_spec: Dict, output_data: Dict, agent_name: str) -> float:
        """Evaluate quality for generic/unknown tasks."""
        score = 0.4  # Base score for any response
        
        if "analysis" in str(output_data):
            score += 0.2
        
        if "comprehensive" in str(output_data):
            score += 0.2
        
        if len(str(output_data)) > 100:  # Substantial content
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_efficiency(self, trajectory: EpisodeTrajectory) -> float:
        """
        Calculate efficiency score based on cost vs benefit.
        
        Args:
            trajectory: Episode trajectory
            
        Returns:
            Efficiency score between 0.0 and 1.0
        """
        if not trajectory.steps:
            return 0.0
        
        total_cost = sum(self._estimate_step_cost(step) for step in trajectory.steps)
        num_steps = len(trajectory.steps)
        
        # More reasonable efficiency calculation
        # Efficiency based on cost efficiency
        if total_cost <= 1.5:
            efficiency = 1.0
        elif total_cost <= 3.0:
            efficiency = 0.8
        elif total_cost <= 4.5:
            efficiency = 0.6
        elif total_cost <= 6.0:
            efficiency = 0.4
        else:
            efficiency = 0.2

        # Less harsh penalty for steps (allow some multi-agent coordination)
        if num_steps > 4:
            efficiency *= 0.9  # Reduced penalty
        elif num_steps > 6:
            efficiency *= 0.8
        
        return efficiency
    
    def _calculate_agent_appropriateness(self, task_specification: str, trajectory: EpisodeTrajectory) -> float:
        """
        Calculate how appropriate the selected agents are for the task.
        
        Args:
            task_specification: Original task specification
            trajectory: Episode trajectory
            
        Returns:
            Appropriateness score between 0.0 and 1.0
        """
        try:
            task_spec = json.loads(task_specification)
        except json.JSONDecodeError:
            task_spec = {"type": "unknown"}
        
        task_type = task_spec.get("type", "unknown")
        urgency = task_spec.get("urgency", "normal")
        detail_level = task_spec.get("detail_level", "summary")
        
        agent_names = [self.agents[step.agent_index].name for step in trajectory.steps]
        appropriateness = 0.0
        
        # Task-specific appropriateness
        if task_type == "search":
            if urgency == "high" and "FastSearchAgent" in agent_names:
                appropriateness += 0.4
            elif urgency != "high" and "DeepAnalysisAgent" in agent_names:
                appropriateness += 0.4
        
        elif task_type == "recommendation":
            if detail_level == "summary" and "FastSearchAgent" in agent_names:
                appropriateness += 0.3
            elif detail_level == "full" and "DeepAnalysisAgent" in agent_names:
                appropriateness += 0.4
        
        elif task_type == "nlp_analysis":
            if "NLPSpecialistAgent" in agent_names:
                appropriateness += 0.5  # High bonus for using specialist
        
        # Bonus for using aggregator when multiple agents were used
        if len(agent_names) > 2 and "ResponseAggregatorAgent" in agent_names:
            appropriateness += 0.3
        
        # Bonus for proper termination
        if "TerminatorAgent" in agent_names:
            appropriateness += 0.2
        
        return min(appropriateness, 1.0)
    
    def _estimate_step_cost(self, step) -> float:
        """Estimate cost for a trajectory step."""
        agent = self.agents[step.agent_index]
        cost_map = {
            "FastSearchAgent": 0.5,
            "DeepAnalysisAgent": 2.0,
            "NLPSpecialistAgent": 1.0,
            "ResponseAggregatorAgent": 0.8,
            "TerminatorAgent": 0.0
        }
        return cost_map.get(agent.name, 1.0)
