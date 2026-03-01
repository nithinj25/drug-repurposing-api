"""
Simple diagnostic test to check API and data sources
"""
import requests
import json

API_URL = "http://localhost:8000"

print("="*70)
print("🔍 Quick API Diagnostic Test")
print("="*70)

# Test 1: Health Check
print("\n1️⃣  Testing Health Endpoint...")
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        print("   ✅ Server is healthy")
        print(f"   Response: {response.json()}")
    else:
        print(f"   ❌ Server returned: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Submit Analysis (async)
print("\n2️⃣  Submitting Analysis Request...")
try:
    payload = {
        "drug_name": "aspirin",
        "indication": "pain"
    }
    
    response = requests.post(
        f"{API_URL}/analyze",
        json=payload,
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print("   ✅ Request submitted successfully")
        print(f"   Job ID: {data.get('job_id')}")
        print(f"   Status: {data.get('status')}")
        
        job_id = data.get('job_id')
        
        # Test 3: Check Job Status
        print("\n3️⃣  Checking Job Status...")
        import time
        time.sleep(3)
        
        status_response = requests.get(f"{API_URL}/jobs/{job_id}", timeout=5)
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"   ✅ Job status: {status_data.get('status')}")
            
            # Check if it's processing
            if status_data.get('status') == 'processing':
                print("   ⏳ Analysis is running in background")
                print("   💡 Check progress at: http://localhost:8000/jobs/" + job_id)
                
            elif status_data.get('status') == 'completed':
                print("   ✅ Analysis complete!")
                
                # Show which agents are using real data
                print("\n4️⃣  Checking Data Sources...")
                tasks = status_data.get('data', {}).get('tasks', {})
                
                for agent_name, task_data in tasks.items():
                    result = task_data.get('result', {})
                    print(f"\n   {agent_name}:")
                    
                    # Check for indicators of real vs mock data
                    result_str = json.dumps(result, default=str)
                    
                    if 'mock' in result_str.lower():
                        print("      ⚠️  Contains mock data indicators")
                    elif 'stub' in result_str.lower():
                        print("      ⚠️  Using stub/rule-based approach")
                    else:
                        print("      ✅ Appears to be using real data processing")
                    
                    # Show method if available
                    if 'method' in result:
                        print(f"      Method: {result['method']}")
                        
        else:
            print(f"   ❌ Status check failed: {status_response.status_code}")
    else:
        print(f"   ❌ Request failed: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("📊 Diagnostic Complete")
print("="*70)
print("\n💡 Tips:")
print("   - If job is 'processing', wait and check: http://localhost:8000/jobs/<job_id>")
print("   - View all jobs: http://localhost:8000/jobs")
print("   - API docs: http://localhost:8000/docs")
