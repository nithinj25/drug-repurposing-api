#!/usr/bin/env python
"""
Simple API Test - Run this while the API server is running
Usage: python test_simple.py
"""

import requests
import json
import time
import sys
from datetime import datetime, timezone

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
            timeout=300
        )
        result = response.json()
        _save_raw_response(result)
        
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
                composite_score = reasoning.get('composite_score')
                decision_level = reasoning.get('decision_level') or reasoning.get('decision')
                hypotheses = reasoning.get('hypotheses') or []

                if composite_score is None and hypotheses:
                    composite_score = hypotheses[0].get('composite_score')
                if decision_level is None and hypotheses:
                    decision_level = hypotheses[0].get('decision')

                if composite_score is not None:
                    print(f"   Composite Score: {composite_score:.2f}")
                else:
                    print("   Composite Score: N/A")

                if decision_level is not None:
                    print(f"   Decision Level: {decision_level}")
                else:
                    print("   Decision Level: N/A")

                if hypotheses:
                    hyp = hypotheses[0]
                    print(f"\n   Top Hypothesis:")
                    print(f"   - Rank: {hyp.get('rank')}")
                    print(f"   - Hypothesis: {hyp.get('hypothesis', 'N/A')[:80]}...")
                    print(f"   - Recommendation: {hyp.get('recommendation', 'N/A')}")

                    print(f"\n   Dimension Scores:")
                    dim_scores = hyp.get('dimension_scores')
                    if isinstance(dim_scores, dict):
                        for dim, score in dim_scores.items():
                            try:
                                print(f"      - {dim:12}: {float(score):.2f}")
                            except (TypeError, ValueError):
                                print(f"      - {dim:12}: {score}")
                    elif isinstance(dim_scores, list):
                        for item in dim_scores:
                            if isinstance(item, dict):
                                dim = item.get('dimension') or item.get('name')
                                score = item.get('score')
                                if dim is not None and score is not None:
                                    print(f"      - {dim:12}: {float(score):.2f}")
                    else:
                        print("      - No dimension scores available")
            
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
            print(f"\n   Raw response saved to: {result.get('_raw_response_path', 'response file')}\n")
            
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


def _save_raw_response(result):
    """Save full API response to a timestamped JSON file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"api_response_{timestamp}.json"
    try:
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        result["_raw_response_path"] = filename
    except Exception as e:
        print(f"⚠️  Could not save raw response: {e}")

if __name__ == "__main__":
    test_api()
