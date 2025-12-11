"""
Test client for Drug Repurposing Assistant API
Demonstrates how to use the API programmatically
"""

import requests
import json
import time
from typing import Dict, Any, List

class DrugRepurposingClient:
    """Client for Drug Repurposing Assistant API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if API is healthy"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def analyze(self, 
                drug_name: str, 
                indication: str,
                drug_synonyms: List[str] = None,
                indication_synonyms: List[str] = None,
                include_patent: bool = True,
                use_internal_data: bool = False) -> Dict[str, Any]:
        """Analyze a single drug-indication pair"""
        
        payload = {
            "drug_name": drug_name,
            "indication": indication,
            "drug_synonyms": drug_synonyms or [],
            "indication_synonyms": indication_synonyms or [],
            "include_patent": include_patent,
            "use_internal_data": use_internal_data
        }
        
        response = self.session.post(
            f"{self.base_url}/analyze",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def batch_analyze(self, 
                     requests_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze multiple drug-indication pairs in batch"""
        
        response = self.session.post(
            f"{self.base_url}/batch",
            json=requests_list
        )
        response.raise_for_status()
        return response.json()
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status and results of a specific job"""
        
        response = self.session.get(f"{self.base_url}/job/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def list_jobs(self) -> Dict[str, Any]:
        """List all processed jobs"""
        
        response = self.session.get(f"{self.base_url}/jobs")
        response.raise_for_status()
        return response.json()
    
    def get_agents_info(self) -> Dict[str, Any]:
        """Get information about available agents"""
        
        response = self.session.get(f"{self.base_url}/agents")
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the session"""
        self.session.close()


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_result(result: Dict[str, Any], indent: int = 0):
    """Pretty print a result"""
    prefix = " " * indent
    
    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}{key}:")
                print_result(value, indent + 2)
            elif isinstance(value, str) and len(value) > 100:
                print(f"{prefix}{key}: {value[:100]}...")
            else:
                print(f"{prefix}{key}: {value}")
    elif isinstance(result, list):
        for i, item in enumerate(result[:3]):  # Show first 3 items
            print(f"{prefix}[{i}]:")
            print_result(item, indent + 2)
        if len(result) > 3:
            print(f"{prefix}... and {len(result) - 3} more items")


def main():
    """Demonstrate API usage"""
    
    print("\n" + "="*70)
    print("  Drug Repurposing Assistant API - Test Client")
    print("="*70)
    
    # Initialize client
    client = DrugRepurposingClient()
    
    try:
        # Test 1: Health check
        print_section("1. Health Check")
        health = client.health_check()
        print(f"✓ Status: {health['status']}")
        print(f"✓ Message: {health['message']}")
        
        # Test 2: Get agents info
        print_section("2. Available Agents")
        agents = client.get_agents_info()
        print(f"✓ Total agents: {agents['total_agents']}")
        for agent_key, agent_info in list(agents['agents'].items())[:3]:
            print(f"  - {agent_info['name']}: {agent_info['description']}")
        print(f"  ... and {agents['total_agents'] - 3} more")
        
        # Test 3: Single analysis
        print_section("3. Single Drug Analysis: Metformin → Cardiovascular Disease")
        print("Submitting request...")
        result = client.analyze(
            drug_name="metformin",
            indication="cardiovascular disease"
        )
        
        if result['success']:
            job_id = result['job_id']
            print(f"✓ Job ID: {job_id}")
            
            # Extract reasoning results
            reasoning = result['data'].get('reasoning_result', {})
            if reasoning:
                print(f"✓ Composite Score: {reasoning.get('composite_score', 'N/A')}")
                print(f"✓ Decision Level: {reasoning.get('decision_level', 'N/A')}")
                print(f"✓ Human Review Required: {result['data'].get('human_review_required', 'N/A')}")
                
                # Show hypothesis
                if reasoning.get('hypotheses'):
                    hyp = reasoning['hypotheses'][0]
                    print(f"\n  Top Hypothesis:")
                    print(f"    - Rank: {hyp.get('rank')}")
                    print(f"    - Hypothesis: {hyp.get('hypothesis', 'N/A')}")
                    print(f"    - Recommendation: {hyp.get('recommendation', 'N/A')}")
                    
                    # Show dimension scores
                    if hyp.get('dimension_scores'):
                        print(f"\n  Dimension Scores:")
                        for dimension, score in hyp['dimension_scores'].items():
                            print(f"    - {dimension.capitalize()}: {score}")
        
        # Test 4: Batch analysis
        print_section("4. Batch Analysis: Multiple Drugs")
        batch_requests = [
            {
                "drug_name": "ibuprofen",
                "indication": "inflammatory bowel disease"
            },
            {
                "drug_name": "aspirin",
                "indication": "diabetes"
            }
        ]
        
        print(f"Submitting {len(batch_requests)} requests...")
        batch_result = client.batch_analyze(batch_requests)
        
        if batch_result['success']:
            print(f"✓ Total processed: {batch_result['total_processed']}")
            for i, job_result in enumerate(batch_result['results'][:2], 1):
                drug = job_result['drug_name']
                indication = job_result['indication']
                job_id = job_result['job_id']
                score = job_result['data']['reasoning_result'].get('composite_score', 'N/A')
                decision = job_result['data']['reasoning_result'].get('decision_level', 'N/A')
                print(f"\n  Result {i}:")
                print(f"    Drug: {drug}")
                print(f"    Indication: {indication}")
                print(f"    Job ID: {job_id}")
                print(f"    Score: {score}")
                print(f"    Decision: {decision}")
        
        # Test 5: List all jobs
        print_section("5. List All Jobs")
        jobs = client.list_jobs()
        print(f"✓ Total jobs processed: {jobs['total_jobs']}")
        if jobs['jobs']:
            print("\n  Recent jobs:")
            for i, job in enumerate(jobs['jobs'][:5], 1):
                print(f"  {i}. {job['drug_name']} → {job['indication']}")
                print(f"     Status: {job['status']}, Tasks: {job['tasks_count']}")
        
        # Test 6: Retrieve specific job
        print_section("6. Retrieve Specific Job Details")
        if result['success']:
            job_id = result['job_id']
            job_details = client.get_job_status(job_id)
            
            if job_details['success']:
                data = job_details['data']
                print(f"✓ Job ID: {data['job_id']}")
                print(f"✓ Drug: {data['drug_name']}")
                print(f"✓ Indication: {data['indication']}")
                print(f"✓ Status: {data['status']}")
                print(f"✓ Total tasks: {len(data['tasks'])}")
                
                # Show task statuses
                print("\n  Task Summary:")
                task_statuses = {}
                for task_id, task in data['tasks'].items():
                    status = task['status']
                    task_statuses[status] = task_statuses.get(status, 0) + 1
                
                for status, count in task_statuses.items():
                    print(f"    - {status}: {count}")
    
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API server")
        print("  Make sure the API is running: python src/api.py")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
    
    finally:
        client.close()
        print_section("Test Complete")
        print("\n✓ API test client finished successfully!\n")


if __name__ == "__main__":
    main()
