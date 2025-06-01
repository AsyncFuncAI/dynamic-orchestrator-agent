#!/usr/bin/env python3
"""
Quick test script for the API Router demo to verify functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from doa_framework import SystemState
from api_router_agents import FastSearchAgent, DeepAnalysisAgent, NLPSpecialistAgent, ResponseAggregatorAgent


def test_agents():
    """Test individual agent functionality."""
    print("🧪 Testing Individual Agents")
    print("=" * 40)
    
    # Test tasks
    search_task = json.dumps({"query": "AI news", "type": "search", "urgency": "high"})
    nlp_task = json.dumps({"text_to_process": "I love this!", "type": "nlp_analysis", "analysis_type": "sentiment"})
    rec_task = json.dumps({"data": {"user_id": 123}, "type": "recommendation", "detail_level": "full"})
    
    agents = [
        FastSearchAgent(),
        DeepAnalysisAgent(),
        NLPSpecialistAgent(),
        ResponseAggregatorAgent()
    ]
    
    tasks = [
        ("Search Task", search_task),
        ("NLP Task", nlp_task),
        ("Recommendation Task", rec_task)
    ]
    
    for task_name, task_spec in tasks:
        print(f"\n📋 {task_name}: {task_spec}")
        state = SystemState(task_spec, [], 0, 5)
        
        for agent in agents:
            try:
                output = agent.execute(state)
                print(f"  • {agent.name}: Cost={output.cost:.1f}, Content={str(output.content)[:100]}...")
            except Exception as e:
                print(f"  • {agent.name}: ERROR - {e}")
    
    # Test aggregator with history
    print(f"\n📋 Testing ResponseAggregator with history:")
    history = [
        ("FastSearchAgent", FastSearchAgent().execute(SystemState(search_task, [], 0, 5))),
        ("DeepAnalysisAgent", DeepAnalysisAgent().execute(SystemState(search_task, [], 0, 5)))
    ]
    state_with_history = SystemState(search_task, history, 2, 5)
    
    aggregator = ResponseAggregatorAgent()
    output = aggregator.execute(state_with_history)
    print(f"  • ResponseAggregator with 2 inputs: Cost={output.cost:.1f}")
    print(f"    Content: {str(output.content)[:150]}...")


def test_task_parsing():
    """Test JSON task specification parsing."""
    print("\n\n🔍 Testing Task Specification Parsing")
    print("=" * 40)
    
    test_tasks = [
        '{"query": "urgent news", "type": "search", "urgency": "high"}',
        '{"text_to_process": "Great product!", "type": "nlp_analysis", "analysis_type": "sentiment"}',
        '{"data": {"user_id": 456}, "type": "recommendation", "detail_level": "summary"}',
        'invalid json string',
        '{"type": "unknown", "data": "test"}'
    ]
    
    for i, task in enumerate(test_tasks):
        print(f"\nTask {i+1}: {task}")
        try:
            parsed = json.loads(task)
            task_type = parsed.get("type", "unknown")
            urgency = parsed.get("urgency", "normal")
            detail = parsed.get("detail_level", "summary")
            analysis = parsed.get("analysis_type", "general")
            print(f"  ✅ Parsed: type={task_type}, urgency={urgency}, detail={detail}, analysis={analysis}")
        except json.JSONDecodeError:
            print(f"  ❌ Invalid JSON - will be treated as unknown type")


if __name__ == "__main__":
    test_agents()
    test_task_parsing()
    print("\n✅ All tests completed!")
