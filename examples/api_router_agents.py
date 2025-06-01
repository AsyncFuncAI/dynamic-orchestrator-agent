#!/usr/bin/env python3
"""
Custom agents for the Self-Optimizing API Router demo.

These agents simulate different microservices in a distributed system,
each with different capabilities, costs, and response times.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import random
from typing import Dict, Any, List
from doa_framework import AgentInterface, SystemState, AgentOutput


class FastSearchAgent(AgentInterface):
    """
    Simulates a fast, keyword-based search service.
    
    Characteristics:
    - Low cost (0.5)
    - Quick response
    - Basic/superficial results
    - Good for high urgency requests
    """
    
    def __init__(self, name: str = "FastSearchAgent"):
        super().__init__(name)
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            task_spec = json.loads(state.task_specification)
        except json.JSONDecodeError:
            task_spec = {"query": state.task_specification, "type": "unknown"}
        
        task_type = task_spec.get("type", "unknown")
        query = task_spec.get("query", "")
        urgency = task_spec.get("urgency", "normal")
        
        if task_type == "search":
            # Simulate fast keyword search
            keywords = query.lower().split()
            results = []
            
            # Basic keyword matching simulation
            if "ai" in keywords or "artificial" in keywords:
                results.append("AI News: Basic overview of recent AI developments")
            if "news" in keywords:
                results.append("Latest Headlines: Top 3 trending topics")
            if "technology" in keywords or "tech" in keywords:
                results.append("Tech Updates: Quick summary of tech news")
            
            if not results:
                results.append(f"Quick search results for: {query}")
            
            content = {
                "service": "FastSearch",
                "results": results[:3],  # Limit to 3 results
                "response_time": "50ms",
                "quality": "basic"
            }
            
            return AgentOutput(
                content=json.dumps(content),
                cost=0.5,
                metadata={
                    "agent_type": "fast_search",
                    "urgency_match": urgency == "high",
                    "result_count": len(results)
                }
            )
        
        elif task_type == "recommendation":
            # Basic recommendation logic
            user_id = task_spec.get("data", {}).get("user_id", "unknown")
            content = {
                "service": "FastRecommendation",
                "recommendations": [
                    f"Popular item for user {user_id}",
                    "Trending recommendation",
                    "Quick suggestion"
                ],
                "confidence": "medium",
                "response_time": "30ms"
            }
            
            return AgentOutput(
                content=json.dumps(content),
                cost=0.5,
                metadata={"agent_type": "fast_recommendation"}
            )
        
        else:
            # Generic fast response
            content = {
                "service": "FastSearch",
                "message": f"Quick processing of {task_type} request",
                "response_time": "25ms"
            }
            
            return AgentOutput(
                content=json.dumps(content),
                cost=0.5,
                metadata={"agent_type": "fast_generic"}
            )


class DeepAnalysisAgent(AgentInterface):
    """
    Simulates a comprehensive, slower analysis service.
    
    Characteristics:
    - High cost (2.0)
    - Detailed, thorough analysis
    - High quality results
    - Takes more time
    """
    
    def __init__(self, name: str = "DeepAnalysisAgent"):
        super().__init__(name)
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            task_spec = json.loads(state.task_specification)
        except json.JSONDecodeError:
            task_spec = {"query": state.task_specification, "type": "unknown"}
        
        task_type = task_spec.get("type", "unknown")
        query = task_spec.get("query", "")
        detail_level = task_spec.get("detail_level", "full")
        
        if task_type == "search":
            # Comprehensive search analysis
            content = {
                "service": "DeepAnalysis",
                "comprehensive_results": [
                    f"Detailed analysis of '{query}' with context",
                    f"Historical trends related to {query}",
                    f"Expert insights and implications",
                    f"Related topics and connections",
                    f"Comprehensive data synthesis"
                ],
                "analysis_depth": "comprehensive",
                "sources_analyzed": 50,
                "confidence_score": 0.92,
                "response_time": "800ms"
            }
            
            return AgentOutput(
                content=json.dumps(content),
                cost=2.0,
                metadata={
                    "agent_type": "deep_analysis",
                    "detail_match": detail_level == "full",
                    "analysis_quality": "high"
                }
            )
        
        elif task_type == "recommendation":
            # Deep recommendation analysis
            user_data = task_spec.get("data", {})
            user_id = user_data.get("user_id", "unknown")
            product_id = user_data.get("product_id", "unknown")
            
            content = {
                "service": "DeepRecommendation",
                "personalized_recommendations": [
                    f"Highly personalized item for user {user_id}",
                    f"Based on behavior analysis and preferences",
                    f"Cross-referenced with similar users",
                    f"Optimized for user engagement",
                    f"Contextual recommendations"
                ],
                "user_profile_analysis": f"Comprehensive profile for user {user_id}",
                "confidence_score": 0.89,
                "personalization_level": "high",
                "response_time": "1200ms"
            }
            
            return AgentOutput(
                content=json.dumps(content),
                cost=2.0,
                metadata={
                    "agent_type": "deep_recommendation",
                    "personalization": "high"
                }
            )
        
        else:
            # Deep analysis for other types
            content = {
                "service": "DeepAnalysis",
                "comprehensive_analysis": f"Thorough analysis of {task_type} request",
                "insights": [
                    "Detailed contextual understanding",
                    "Multi-dimensional analysis",
                    "Predictive insights"
                ],
                "confidence_score": 0.85,
                "response_time": "950ms"
            }
            
            return AgentOutput(
                content=json.dumps(content),
                cost=2.0,
                metadata={"agent_type": "deep_generic"}
            )


class NLPSpecialistAgent(AgentInterface):
    """
    Simulates a specialized NLP processing service.
    
    Characteristics:
    - Medium cost (1.0)
    - Specialized for NLP tasks
    - High quality for NLP, low relevance for other tasks
    - Specific analysis capabilities
    """
    
    def __init__(self, name: str = "NLPSpecialistAgent"):
        super().__init__(name)
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            task_spec = json.loads(state.task_specification)
        except json.JSONDecodeError:
            task_spec = {"text_to_process": state.task_specification, "type": "unknown"}
        
        task_type = task_spec.get("type", "unknown")
        
        if task_type == "nlp_analysis":
            text_to_process = task_spec.get("text_to_process", "")
            analysis_type = task_spec.get("analysis_type", "general")
            
            # Specialized NLP processing
            if analysis_type == "sentiment":
                content = {
                    "service": "NLPSpecialist",
                    "analysis_type": "sentiment",
                    "sentiment_score": random.uniform(-1, 1),
                    "sentiment_label": random.choice(["positive", "negative", "neutral"]),
                    "confidence": random.uniform(0.8, 0.95),
                    "text_analyzed": text_to_process,
                    "response_time": "200ms"
                }
            elif analysis_type == "entity_extraction":
                content = {
                    "service": "NLPSpecialist",
                    "analysis_type": "entity_extraction",
                    "entities": [
                        {"text": "example entity", "type": "PERSON", "confidence": 0.9},
                        {"text": "another entity", "type": "ORG", "confidence": 0.85}
                    ],
                    "entity_count": 2,
                    "response_time": "180ms"
                }
            else:
                content = {
                    "service": "NLPSpecialist",
                    "analysis_type": "general",
                    "language_detected": "en",
                    "text_complexity": "medium",
                    "key_phrases": ["important phrase", "key concept"],
                    "response_time": "150ms"
                }
            
            return AgentOutput(
                content=json.dumps(content),
                cost=1.0,
                metadata={
                    "agent_type": "nlp_specialist",
                    "specialization_match": True,
                    "analysis_type": analysis_type
                }
            )
        
        else:
            # Low relevance for non-NLP tasks
            content = {
                "service": "NLPSpecialist",
                "message": f"NLP service not optimized for {task_type} tasks",
                "relevance": "low",
                "response_time": "50ms"
            }
            
            return AgentOutput(
                content=json.dumps(content),
                cost=1.0,
                metadata={
                    "agent_type": "nlp_specialist",
                    "specialization_match": False,
                    "relevance": "low"
                }
            )


class ResponseAggregatorAgent(AgentInterface):
    """
    Simulates a service that aggregates and synthesizes responses from other services.
    
    Characteristics:
    - Medium cost (0.8)
    - Improves overall response quality
    - Depends on quality of previous outputs
    - Provides coherent synthesis
    """
    
    def __init__(self, name: str = "ResponseAggregatorAgent"):
        super().__init__(name)
    
    def execute(self, state: SystemState) -> AgentOutput:
        if not state.history:
            return AgentOutput(
                content=json.dumps({
                    "service": "ResponseAggregator",
                    "message": "No previous responses to aggregate",
                    "response_time": "10ms"
                }),
                cost=0.8,
                metadata={"agent_type": "aggregator", "input_count": 0}
            )
        
        # Analyze previous outputs
        previous_outputs = []
        total_quality_indicators = 0
        
        for agent_name, output in state.history:
            if agent_name != "ResponseAggregatorAgent":
                try:
                    output_data = json.loads(str(output.content))
                    previous_outputs.append({
                        "agent": agent_name,
                        "data": output_data,
                        "cost": output.cost
                    })
                    
                    # Quality indicators
                    if "confidence" in str(output.content):
                        total_quality_indicators += 1
                    if "comprehensive" in str(output.content):
                        total_quality_indicators += 1
                    if "analysis" in str(output.content):
                        total_quality_indicators += 1
                        
                except (json.JSONDecodeError, TypeError):
                    previous_outputs.append({
                        "agent": agent_name,
                        "data": {"content": str(output.content)},
                        "cost": output.cost
                    })
        
        # Synthesize response
        synthesis_quality = min(total_quality_indicators / max(len(previous_outputs), 1), 1.0)
        
        content = {
            "service": "ResponseAggregator",
            "synthesis": "Aggregated response from multiple services",
            "input_services": [output["agent"] for output in previous_outputs],
            "synthesis_quality": synthesis_quality,
            "total_inputs": len(previous_outputs),
            "combined_insights": [
                "Synthesized key findings",
                "Cross-service validation",
                "Coherent response compilation"
            ],
            "response_time": "120ms"
        }
        
        return AgentOutput(
            content=json.dumps(content),
            cost=0.8,
            metadata={
                "agent_type": "aggregator",
                "input_count": len(previous_outputs),
                "synthesis_quality": synthesis_quality
            }
        )
