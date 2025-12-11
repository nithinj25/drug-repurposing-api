#!/usr/bin/env python
"""
Complete API Output Test - Shows ALL response data
Usage: python test_complete_output.py
"""

import requests
import json
import time
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_complete():
    """Test API and show COMPLETE response"""
    
    base_url = "http://localhost:8000"
    
    print_section("DRUG REPURPOSING API - COMPLETE OUTPUT TEST")
    
    # Test 1: Health Check
    print("\n[1/3] Testing API Health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print("[OK] API is running")
        print(f"     Status: {response.json()['status']}")
    except Exception as e:
        print(f"[FAIL] API not responding: {str(e)}")
        print("\n     Start the API server first:")
        print("     python src/api.py\n")
        return
    
    # Test 2: Get Agents
    print("\n[2/3] Getting Available Agents...")
    try:
        response = requests.get(f"{base_url}/agents", timeout=10)
        agents = response.json()
        print(f"[OK] Found {agents['total_agents']} specialized agents")
        for name, info in agents['agents'].items():
            print(f"     - {info['name']}: {info['dimension']}")
    except Exception as e:
        print(f"[FAIL] Error: {str(e)}")
        return
    
    # Test 3: COMPLETE Drug Analysis
    print_section("COMPLETE DRUG ANALYSIS - ALL DATA")
    
    print("\nAnalyzing: Aspirin for Cancer Prevention")
    print("This will take 10-30 seconds...\n")
    
    try:
        payload = {
            "drug_name": "aspirin",
            "indication": "cancer prevention"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/analyze",
            json=payload,
            timeout=180
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"[OK] Analysis completed in {elapsed:.1f} seconds")
            print(f"\n{'='*80}")
            print("JOB METADATA")
            print(f"{'='*80}")
            print(f"Job ID:     {result['job_id']}")
            print(f"Drug:       {result['data']['drug_name']}")
            print(f"Indication: {result['data']['indication']}")
            print(f"Status:     {result['data']['status']}")
            print(f"Created:    {result['data'].get('created_at', 'N/A')}")
            
            # REASONING RESULTS
            if 'reasoning_result' in result['data']:
                reasoning = result['data']['reasoning_result']
                
                print(f"\n{'='*80}")
                print("REASONING & SCORING")
                print(f"{'='*80}")
                print(f"Decision Level:   {reasoning.get('decision_level', 'N/A')}")
                print(f"Composite Score:  {reasoning.get('composite_score', 'N/A')}")
                print(f"Processing Time:  {reasoning.get('processing_time_ms', 'N/A')} ms")
                
                # Dimension Scores
                if 'dimension_scores' in reasoning:
                    print(f"\nDimension Scores:")
                    for dim, score in reasoning['dimension_scores'].items():
                        bar = '#' * int(score * 20) if isinstance(score, (int, float)) else ''
                        print(f"  {dim:15s}: {score:6s} {bar}")
                
                # Hypotheses
                if 'hypotheses' in reasoning:
                    print(f"\nHypotheses Generated: {len(reasoning['hypotheses'])}")
                    for idx, hyp in enumerate(reasoning['hypotheses'], 1):
                        print(f"\n  Hypothesis #{idx}:")
                        print(f"    Drug:         {hyp.get('drug', 'N/A')}")
                        print(f"    Disease:      {hyp.get('disease', 'N/A')}")
                        print(f"    Score:        {hyp.get('score', 'N/A')}")
                        print(f"    Evidence:     {len(hyp.get('supporting_evidence', []))} items")
                        print(f"    Confidence:   {hyp.get('confidence', 'N/A')}")
                        
                        if 'explanation' in hyp:
                            print(f"    Explanation:")
                            explanation = hyp['explanation']
                            # Word wrap explanation
                            words = explanation.split()
                            line = "      "
                            for word in words:
                                if len(line) + len(word) > 76:
                                    print(line)
                                    line = "      " + word
                                else:
                                    line += " " + word if len(line) > 6 else word
                            if line.strip():
                                print(line)
            
            # AGENT TASK RESULTS (COMPLETE)
            if 'tasks' in result['data']:
                print(f"\n{'='*80}")
                print(f"AGENT TASK RESULTS - ALL {len(result['data']['tasks'])} AGENTS")
                print(f"{'='*80}")
                
                for task_id, task in result['data']['tasks'].items():
                    agent_name = task.get('agent_name', 'Unknown')
                    dimension = task.get('dimension', 'Unknown')
                    status = task.get('status', 'Unknown')
                    
                    print(f"\n[{agent_name.upper()}] - {dimension.title()}")
                    print(f"  Task ID: {task_id}")
                    print(f"  Status:  {status}")
                    
                    if 'started_at' in task:
                        print(f"  Started: {task['started_at']}")
                    if 'completed_at' in task:
                        print(f"  Completed: {task['completed_at']}")
                    
                    # Extract and display result details
                    if 'result' in task and task['result']:
                        task_result = task['result']
                        
                        if isinstance(task_result, dict):
                            # Summary
                            if 'summary' in task_result:
                                print(f"  Summary: {str(task_result['summary'])[:100]}...")
                            
                            # Count evidence items
                            evidence_fields = {
                                'papers': 'Literature Papers',
                                'trials': 'Clinical Trials',
                                'patents': 'Patents',
                                'evidence': 'Evidence Items',
                                'market_data': 'Market Data',
                                'safety_data': 'Safety Data'
                            }
                            
                            for field, label in evidence_fields.items():
                                if field in task_result:
                                    if isinstance(task_result[field], list):
                                        count = len(task_result[field])
                                        if count > 0:
                                            print(f"  {label}: {count} items")
                                    elif isinstance(task_result[field], dict):
                                        print(f"  {label}: Available")
                            
                            # Display key metrics
                            metric_fields = {
                                'safety_score': 'Safety Score',
                                'feasibility_score': 'Feasibility Score',
                                'fto_score': 'FTO Score',
                                'risk_level': 'Risk Level',
                                'tam': 'TAM',
                                'cagr': 'CAGR',
                                'competition_level': 'Competition'
                            }
                            
                            for field, label in metric_fields.items():
                                if field in task_result:
                                    print(f"  {label}: {task_result[field]}")
                            
                            # Show nested structure summary
                            if len(task_result) > 10:
                                print(f"  Total fields in result: {len(task_result)}")
                    
                    if 'error' in task and task['error']:
                        print(f"  ERROR: {task['error']}")
            
            # COMPLETE JSON OUTPUT
            print(f"\n{'='*80}")
            print("COMPLETE JSON RESPONSE")
            print(f"{'='*80}")
            
            json_output = json.dumps(result, indent=2, ensure_ascii=False)
            print(f"\nTotal size: {len(json_output)} characters")
            print(f"Total lines: {len(json_output.splitlines())}")
            print(f"\n{json_output}")
            
            # SUCCESS SUMMARY
            print(f"\n{'='*80}")
            print("TEST COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"\nAll data retrieved successfully!")
            print(f"  - Reasoning result: {'Yes' if 'reasoning_result' in result['data'] else 'No'}")
            print(f"  - Agent tasks: {len(result['data'].get('tasks', {}))} agents")
            print(f"  - Total response size: {len(json_output)} characters")
            
        else:
            print(f"[FAIL] Analysis failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"[FAIL] Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete()
