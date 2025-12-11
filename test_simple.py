#!/usr/bin/env python
"""
Simple API Test - Run this while the API server is running
Usage: python test_simple.py
"""

import requests
import json
import time
import sys

def test_api():
    """Run simple API tests"""
    
    base_url = "http://localhost:8000"
    
    print("\n" + "="*70)
    print("  SIMPLE API TEST")
    print("="*70)
    
    # Test 1: Health Check
    print("\n[1/4] Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print("✓ API is running!")
        print(f"   Status: {response.json()['status']}")
    except Exception as e:
        print(f"✗ API not responding: {str(e)}")
        print("   Make sure to run: python src/api.py")
        return False
    
    # Test 2: Get Agents Info
    print("\n[2/4] Getting Available Agents...")
    try:
        response = requests.get(f"{base_url}/agents", timeout=10)
        agents = response.json()
        print(f"✓ Found {agents['total_agents']} agents")
        for name, info in list(agents['agents'].items())[:3]:
            print(f"   - {info['name']}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False
    
    # Test 3: Single Drug Analysis
    print("\n[3/4] Analyzing Drug: Metformin → Cardiovascular Disease...")
    print("   (This will take 5-15 seconds)")
    try:
        payload = {
            "drug_name": "metformin",
            "indication": "cardiovascular disease"
        }
        response = requests.post(
            f"{base_url}/analyze",
            json=payload,
            timeout=120
        )
        result = response.json()
        
        if result.get('success'):
            data = result['data']
            reasoning = data.get('reasoning_result')
            
            print(f"✓ Analysis Complete!")
            print(f"\n   Job ID: {data['job_id']}")
            print(f"   Drug: {data['drug_name']}")
            print(f"   Indication: {data['indication']}")
            print(f"   Status: {data['status']}")
            
            if reasoning:
                print(f"\n   === REASONING RESULTS ===")
                print(f"   Composite Score: {reasoning['composite_score']:.2f}")
                print(f"   Decision Level: {reasoning['decision_level']}")
                
                if reasoning.get('hypotheses'):
                    hyp = reasoning['hypotheses'][0]
                    print(f"\n   Top Hypothesis:")
                    print(f"   - Rank: {hyp.get('rank')}")
                    print(f"   - Hypothesis: {hyp.get('hypothesis', 'N/A')[:80]}...")
                    print(f"   - Recommendation: {hyp.get('recommendation', 'N/A')}")
                    
                    print(f"\n   Dimension Scores:")
                    for dim, score in hyp.get('dimension_scores', {}).items():
                        print(f"      - {dim:12}: {score:.2f}")
            
            # Print all agent task results
            if data.get('tasks'):
                print(f"\n   === AGENT RESULTS ===")
                for task_id, task in data['tasks'].items():
                    agent = task.get('agent_name', 'Unknown')
                    status = task.get('status', 'Unknown')
                    print(f"   {agent}: {status}")
                    
                    if task.get('result'):
                        result_data = task['result']
                        if isinstance(result_data, dict):
                            print(f"      Evidence count: {result_data.get('evidence_count', 'N/A')}")
                            if result_data.get('summary'):
                                print(f"      Summary: {result_data['summary'][:60]}...")
            
            print(f"\n   === FULL RESPONSE (JSON) ===")
            print(json.dumps(result, indent=2)[:2000] + "\n   ... (truncated for display)")
            
        else:
            print(f"✗ Analysis failed")
            print(json.dumps(result, indent=2))
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: List Jobs
    print("\n[4/4] Listing All Jobs...")
    try:
        response = requests.get(f"{base_url}/jobs", timeout=10)
        jobs = response.json()
        print(f"✓ Total jobs processed: {jobs['total_jobs']}")
        if jobs['jobs']:
            for i, job in enumerate(jobs['jobs'][:3], 1):
                print(f"   {i}. {job['drug_name']} → {job['indication']} ({job['status']})")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False
    
    print("\n" + "="*70)
    print("  ALL TESTS COMPLETED SUCCESSFULLY! ✓")
    print("="*70 + "\n")
    return True

if __name__ == "__main__":
    test_api()
