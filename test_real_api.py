"""
Test script to verify real API calls vs mock data
"""
import requests
import json
import time

API_URL = "http://localhost:8000"

def test_analyze_endpoint():
    """Test the analyze endpoint with a real drug"""
    print("="*70)
    print("🧪 Testing Drug Repurposing Assistant - Real API vs Mock Data")
    print("="*70)
    
    # Test with a well-known drug
    payload = {
        "drug_name": "aspirin",
        "indication": "cardiovascular disease",
        "query": "antiplatelet effects"
    }
    
    print("\n📤 Submitting analysis request...")
    print(f"Drug: {payload['drug_name']}")
    print(f"Indication: {payload['indication']}")
    print(f"Query: {payload['query']}")
    
    try:
        # Submit job
        response = requests.post(f"{API_URL}/analyze", json=payload, timeout=120)
        
        if response.status_code != 200:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            return
        
        data = response.json()
        job_id = data.get("job_id")
        
        print(f"\n✅ Job submitted successfully!")
        print(f"Job ID: {job_id}")
        print(f"Status: {data.get('status')}")
        
        # Poll for results
        print("\n⏳ Waiting for results...")
        max_attempts = 60
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(2)
            attempt += 1
            
            status_response = requests.get(f"{API_URL}/jobs/{job_id}", timeout=10)
            
            if status_response.status_code == 200:
                result = status_response.json()
                status = result.get("status")
                
                print(f"   Attempt {attempt}/60 - Status: {status}")
                
                if status == "completed":
                    print("\n" + "="*70)
                    print("✅ ANALYSIS COMPLETE - Checking Data Sources")
                    print("="*70)
                    
                    # Analyze results to see if real data was used
                    analyze_results(result)
                    return result
                
                elif status == "failed":
                    print(f"\n❌ Analysis failed: {result.get('error')}")
                    return None
        
        print("\n⏰ Timeout waiting for results")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Make sure the server is running: python src/api.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

def analyze_results(result):
    """Analyze results to determine if real APIs were used"""
    
    print("\n📊 RESULTS ANALYSIS:")
    print("-" * 70)
    
    data = result.get("data", result)
    tasks = data.get("tasks", {})
    
    # Check Literature Agent
    if "literature_agent" in tasks:
        lit_result = tasks["literature_agent"].get("result", {})
        papers = lit_result.get("papers", [])
        papers_found = lit_result.get("papers_found", 0)
        
        print("\n📚 Literature Agent:")
        if papers and len(papers) > 0:
            paper = papers[0]
            # Check if it's mock data (mock data has generic titles)
            if "Mock" in paper.get("title", "") or "Example" in paper.get("title", ""):
                print("   ⚠️  Using MOCK DATA")
                print(f"   Title: {paper.get('title', 'N/A')}")
            else:
                print("   ✅ Using REAL DATA from PubMed")
                print(f"   Papers found: {papers_found}")
                print(f"   First paper: {paper.get('title', 'N/A')[:80]}...")
        else:
            print("   ⚠️  No papers found (may be using mock data)")
    
    # Check Clinical Agent
    if "clinical_agent" in tasks:
        clin_result = tasks["clinical_agent"].get("result", {})
        trials = clin_result.get("trials", [])
        trials_found = clin_result.get("trials_found", 0)
        
        print("\n🏥 Clinical Agent:")
        if trials and len(trials) > 0:
            trial = trials[0]
            nct_id = trial.get("nct_id", "")
            # Check if it's mock data
            if "MOCK" in nct_id or not nct_id.startswith("NCT"):
                print("   ⚠️  Using MOCK DATA")
                print(f"   Trial ID: {nct_id}")
            else:
                print("   ✅ Using REAL DATA from ClinicalTrials.gov")
                print(f"   Trials found: {trials_found}")
                print(f"   First trial: {nct_id} - {trial.get('title', 'N/A')[:60]}...")
        else:
            print("   ⚠️  No trials found (may be using mock data)")
    
    # Check Molecular Agent
    if "molecular_agent" in tasks:
        mol_result = tasks["molecular_agent"].get("result", {})
        method = mol_result.get("method", "")
        targets = mol_result.get("predicted_targets", [])
        
        print("\n🧬 Molecular Agent:")
        if "stub" in method or "rule_based" in method:
            print("   ⚠️  Using RULE-BASED STUB (not real API)")
            print(f"   Method: {method}")
        else:
            print("   ✅ Using advanced molecular analysis")
        print(f"   Targets found: {targets}")
    
    # Check Safety Agent
    if "safety_agent" in tasks:
        safety_result = tasks["safety_agent"].get("result", {})
        adverse_events = safety_result.get("adverse_events", [])
        
        print("\n⚠️  Safety Agent:")
        if adverse_events and len(adverse_events) > 0:
            ae = adverse_events[0]
            source = ae.get("source", "unknown")
            if source == "unknown" or not source:
                print("   ⚠️  May be using mock data")
            else:
                print(f"   ✅ Using REAL DATA from: {source}")
                print(f"   Adverse events found: {len(adverse_events)}")
        else:
            print("   ℹ️  No adverse events found")
    
    # Check Market Agent
    if "market_agent" in tasks:
        market_result = tasks["market_agent"].get("result", {})
        
        print("\n💼 Market Agent:")
        print("   ⚠️  Using MOCK DATA (IQVIA/Bloomberg not available)")
        print("   Note: Market APIs require enterprise subscriptions")
    
    # Check Patent Agent
    if "patent_agent" in tasks:
        patent_result = tasks["patent_agent"].get("result", {})
        patents = patent_result.get("patents", [])
        
        print("\n📋 Patent Agent:")
        if patents and len(patents) > 0:
            patent = patents[0]
            patent_id = patent.get("patent_id", "")
            if "MOCK" in patent_id or not patent_id:
                print("   ⚠️  Using MOCK DATA")
            else:
                print("   ✅ Using REAL DATA from PatentsView")
                print(f"   Patents found: {len(patents)}")
        else:
            print("   ⚠️  No patents found")
    
    # Check Reasoning Agent
    if "reasoning_result" in data:
        reasoning = data["reasoning_result"]
        hypotheses = reasoning.get("hypotheses", [])
        
        print("\n🧠 Reasoning Agent:")
        if hypotheses and len(hypotheses) > 0:
            print("   ✅ AI reasoning completed")
            print(f"   Hypotheses generated: {len(hypotheses)}")
            if len(hypotheses) > 0:
                print(f"   First hypothesis: {hypotheses[0][:80]}...")
        else:
            print("   ⚠️  No reasoning output")
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print("✅ = Real API data being used")
    print("⚠️  = Mock/synthetic data or no data")
    print("="*70)

if __name__ == "__main__":
    result = test_analyze_endpoint()
    
    # Save full results to file
    if result:
        with open("test_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print("\n💾 Full results saved to: test_results.json")
