#!/usr/bin/env python
"""
Drug Repurposing Assistant API - Complete Demo Example
Shows how to use the API with real examples
Run this script while the API server is running: python src/api.py
"""

import requests
import json
import time
from typing import Dict, Any

# API Base URL
API_URL = "http://localhost:8000"

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def print_success(message: str):
    """Print success message"""
    print(f"✓ {message}")

def print_error(message: str):
    """Print error message"""
    print(f"✗ {message}")

def print_json(data: Dict[str, Any], indent: int = 2):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=indent))

# ==============================================================================
# EXAMPLE 1: Health Check
# ==============================================================================

def example_1_health_check():
    """Test if API is running"""
    print_section("EXAMPLE 1: Health Check")
    print("Code:")
    print("""
    import requests
    response = requests.get("http://localhost:8000/health")
    print(response.json())
    """)
    print("\nExecuting...\n")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        
        print_success(f"Status Code: {response.status_code}")
        print("Response:")
        print_json(response.json())
        return True
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to API server")
        print("Make sure to run: python src/api.py")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ==============================================================================
# EXAMPLE 2: Single Drug Analysis
# ==============================================================================

def example_2_single_analysis():
    """Analyze a single drug-indication pair"""
    print_section("EXAMPLE 2: Analyze Single Drug")
    print("Code:")
    print("""
    import requests
    
    payload = {
        "drug_name": "metformin",
        "indication": "cardiovascular disease"
    }
    
    response = requests.post(
        "http://localhost:8000/analyze",
        json=payload
    )
    result = response.json()
    print(f"Composite Score: {result['data']['reasoning_result']['composite_score']}")
    print(f"Decision: {result['data']['reasoning_result']['decision_level']}")
    """)
    print("\nExecuting...\n")
    
    try:
        payload = {
            "drug_name": "metformin",
            "indication": "cardiovascular disease"
        }
        
        print("Sending request...")
        print(f"  Drug: {payload['drug_name']}")
        print(f"  Indication: {payload['indication']}")
        print("\nWaiting for analysis (5-15 seconds)...\n")
        
        response = requests.post(
            f"{API_URL}/analyze",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        
        if result.get('success'):
            data = result['data']
            job_id = data['job_id']
            
            print_success(f"Analysis Complete!")
            print(f"\nJob ID: {job_id}")
            print(f"Drug: {data['drug_name']}")
            print(f"Indication: {data['indication']}")
            print(f"Status: {data['status']}")
            
            if data.get('reasoning_result'):
                reasoning = data['reasoning_result']
                print(f"\n--- Reasoning Results ---")
                print(f"Composite Score: {reasoning.get('composite_score', 'N/A')}")
                print(f"Decision Level: {reasoning.get('decision_level', 'N/A')}")
                
                if reasoning.get('hypotheses'):
                    hyp = reasoning['hypotheses'][0]
                    print(f"\nTop Hypothesis:")
                    print(f"  Rank: {hyp.get('rank')}")
                    print(f"  Hypothesis: {hyp.get('hypothesis', 'N/A')[:100]}...")
                    print(f"  Recommendation: {hyp.get('recommendation', 'N/A')}")
                    
                    print(f"\nDimension Scores:")
                    for dim, score in hyp.get('dimension_scores', {}).items():
                        print(f"  - {dim.capitalize():15} : {score:.2f}")
                
                print(f"\n--- Full Response ---")
                print_json(result, indent=2)
            
            return True
        else:
            print_error("Analysis failed")
            print_json(result)
            return False
            
    except requests.exceptions.Timeout:
        print_error("Request timed out (took too long)")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ==============================================================================
# EXAMPLE 3: List Available Agents
# ==============================================================================

def example_3_list_agents():
    """Get information about available agents"""
    print_section("EXAMPLE 3: List Available Agents")
    print("Code:")
    print("""
    import requests
    response = requests.get("http://localhost:8000/agents")
    agents = response.json()
    for agent_name, agent_info in agents['agents'].items():
        print(f"{agent_info['name']}: {agent_info['description']}")
    """)
    print("\nExecuting...\n")
    
    try:
        response = requests.get(f"{API_URL}/agents", timeout=10)
        response.raise_for_status()
        
        agents = response.json()
        print_success(f"Found {agents['total_agents']} agents\n")
        
        for agent_key, agent_info in agents['agents'].items():
            print(f"Agent: {agent_info['name']}")
            print(f"  Description: {agent_info['description']}")
            print(f"  Sources: {', '.join(agent_info['sources'])}")
            print(f"  Dimension: {agent_info['dimension']}")
            print()
        
        return True
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ==============================================================================
# EXAMPLE 4: List All Jobs
# ==============================================================================

def example_4_list_jobs():
    """List all processed jobs"""
    print_section("EXAMPLE 4: List All Jobs")
    print("Code:")
    print("""
    import requests
    response = requests.get("http://localhost:8000/jobs")
    jobs = response.json()
    for job in jobs['jobs']:
        print(f"{job['drug_name']} -> {job['indication']}: {job['status']}")
    """)
    print("\nExecuting...\n")
    
    try:
        response = requests.get(f"{API_URL}/jobs", timeout=10)
        response.raise_for_status()
        
        jobs = response.json()
        total = jobs['total_jobs']
        
        print_success(f"Total jobs processed: {total}\n")
        
        if jobs['jobs']:
            print("Recent Jobs:")
            for i, job in enumerate(jobs['jobs'][:5], 1):
                print(f"{i}. {job['drug_name']} → {job['indication']}")
                print(f"   Status: {job['status']}, Tasks: {job['tasks_count']}")
                print(f"   Created: {job['created_at']}")
                print()
        else:
            print("No jobs processed yet.")
        
        return True
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ==============================================================================
# EXAMPLE 5: Batch Analysis
# ==============================================================================

def example_5_batch_analysis():
    """Analyze multiple drug-indication pairs at once"""
    print_section("EXAMPLE 5: Batch Analysis (Multiple Drugs)")
    print("Code:")
    print("""
    import requests
    
    drugs = [
        {"drug_name": "aspirin", "indication": "diabetes"},
        {"drug_name": "ibuprofen", "indication": "inflammatory bowel disease"}
    ]
    
    response = requests.post("http://localhost:8000/batch", json=drugs)
    results = response.json()
    
    for result in results['results']:
        print(f"{result['drug_name']}: {result['data']['reasoning_result']['composite_score']}")
    """)
    print("\nExecuting...\n")
    
    try:
        batch_requests = [
            {"drug_name": "aspirin", "indication": "diabetes"},
            {"drug_name": "ibuprofen", "indication": "inflammatory bowel disease"}
        ]
        
        print(f"Submitting {len(batch_requests)} drug-indication pairs...")
        for req in batch_requests:
            print(f"  - {req['drug_name']} for {req['indication']}")
        
        print("\nWaiting for batch analysis (10-30 seconds)...\n")
        
        response = requests.post(
            f"{API_URL}/batch",
            json=batch_requests,
            timeout=180
        )
        response.raise_for_status()
        
        results = response.json()
        
        if results.get('success'):
            print_success(f"Batch analysis complete!")
            print(f"Total processed: {results['total_processed']}\n")
            
            for i, job_result in enumerate(results['results'], 1):
                print(f"Result {i}:")
                print(f"  Drug: {job_result['drug_name']}")
                print(f"  Indication: {job_result['indication']}")
                print(f"  Job ID: {job_result['job_id']}")
                
                if job_result['data'].get('reasoning_result'):
                    reasoning = job_result['data']['reasoning_result']
                    print(f"  Composite Score: {reasoning.get('composite_score', 'N/A')}")
                    print(f"  Decision: {reasoning.get('decision_level', 'N/A')}")
                print()
            
            return True
        else:
            print_error("Batch analysis failed")
            return False
            
    except requests.exceptions.Timeout:
        print_error("Request timed out")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ==============================================================================
# EXAMPLE 6: Get Specific Job Result
# ==============================================================================

def example_6_get_job_result(job_id: str = None):
    """Retrieve results for a specific job"""
    print_section("EXAMPLE 6: Get Specific Job Result")
    print("Code:")
    print("""
    import requests
    
    job_id = "your-job-id-here"
    response = requests.get(f"http://localhost:8000/job/{job_id}")
    result = response.json()
    print(result['data']['reasoning_result'])
    """)
    print("\nExecuting...\n")
    
    # Get a job ID from the list of jobs
    try:
        response = requests.get(f"{API_URL}/jobs", timeout=10)
        jobs = response.json()
        
        if not jobs['jobs']:
            print("No jobs available yet. Run examples 2 or 5 first.")
            return False
        
        job_id = jobs['jobs'][0]['job_id']
        print(f"Retrieving job: {job_id}\n")
        
        response = requests.get(f"{API_URL}/job/{job_id}", timeout=10)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get('success'):
            data = result['data']
            print_success("Job retrieved successfully!\n")
            print(f"Drug: {data['drug_name']}")
            print(f"Indication: {data['indication']}")
            print(f"Status: {data['status']}")
            
            print("\n--- Full Result ---")
            print_json(result, indent=2)
            return True
        else:
            print_error("Failed to retrieve job")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ==============================================================================
# MAIN MENU
# ==============================================================================

def main_menu():
    """Interactive menu for running examples"""
    print("\n" + "=" * 70)
    print("  DRUG REPURPOSING ASSISTANT - API EXAMPLES")
    print("=" * 70)
    print("""
Make sure the API server is running first:
    python src/api.py

Then choose an example to run:
    1. Health Check (fastest)
    2. Analyze Single Drug (5-15 seconds)
    3. List Available Agents
    4. List All Jobs
    5. Batch Analysis (Multiple Drugs, 10-30 seconds)
    6. Get Specific Job Result
    7. Run All Examples
    0. Exit

Note: Examples 2 and 5 will take 5-30 seconds as they analyze real data
from multiple external APIs (PubMed, ClinicalTrials.gov, USPTO, etc.)
""")

def run_example(choice: int):
    """Run the selected example"""
    examples = {
        1: ("Health Check", example_1_health_check),
        2: ("Single Drug Analysis", example_2_single_analysis),
        3: ("List Agents", example_3_list_agents),
        4: ("List Jobs", example_4_list_jobs),
        5: ("Batch Analysis", example_5_batch_analysis),
        6: ("Get Job Result", example_6_get_job_result),
    }
    
    if choice in examples:
        name, func = examples[choice]
        try:
            success = func()
            print("\n" + "=" * 70)
            if success:
                print(f"✓ {name} - COMPLETED SUCCESSFULLY")
            else:
                print(f"✗ {name} - FAILED")
            print("=" * 70 + "\n")
            return success
        except KeyboardInterrupt:
            print("\n\nExample interrupted by user.")
            return False
        except Exception as e:
            print_error(f"Unexpected error: {str(e)}")
            return False
    else:
        print_error("Invalid choice")
        return False

def run_all_examples():
    """Run all examples in sequence"""
    print_section("Running All Examples")
    
    results = []
    
    # Example 1: Health Check
    print("Running Example 1: Health Check")
    if not run_example(1):
        print_error("API server is not running. Please start it with: python src/api.py")
        return
    
    time.sleep(1)
    
    # Example 3: List Agents
    print("Running Example 3: List Agents")
    run_example(3)
    time.sleep(1)
    
    # Example 2: Single Analysis
    print("Running Example 2: Single Drug Analysis")
    run_example(2)
    time.sleep(2)
    
    # Example 4: List Jobs
    print("Running Example 4: List Jobs")
    run_example(4)
    time.sleep(1)
    
    # Example 5: Batch Analysis
    print("Running Example 5: Batch Analysis")
    run_example(5)
    time.sleep(2)
    
    # Example 6: Get Job Result
    print("Running Example 6: Get Job Result")
    run_example(6)
    
    print("\n" + "=" * 70)
    print("  ALL EXAMPLES COMPLETED!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Allow running specific example from command line
        try:
            choice = int(sys.argv[1])
            if choice == 7:
                run_all_examples()
            elif choice == 0:
                print("Exiting...")
            else:
                run_example(choice)
        except ValueError:
            print_error("Invalid argument. Use: python run_api_examples.py [1-7]")
    else:
        # Interactive mode
        main_menu()
        
        while True:
            try:
                choice = input("Enter your choice (0-7): ").strip()
                
                if not choice:
                    continue
                
                try:
                    choice = int(choice)
                except ValueError:
                    print_error("Please enter a number between 0-7")
                    continue
                
                if choice == 0:
                    print("\nExiting...")
                    break
                elif choice == 7:
                    run_all_examples()
                else:
                    run_example(choice)
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print_error(f"Error: {str(e)}")
