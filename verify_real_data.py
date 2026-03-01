"""
Comprehensive test to verify if the system is fetching REAL data or using MOCK data
We'll submit a query and inspect the actual results in detail
"""
import requests
import json
import time
import re

API_URL = "http://localhost:8000"

def check_if_mock_data(text):
    """Check for common mock data indicators"""
    mock_indicators = [
        'mock', 'Mock', 'MOCK',
        'example', 'Example', 'EXAMPLE',
        'dummy', 'Dummy', 'DUMMY',
        'test', 'Test', 'placeholder',
        'lorem ipsum', 'sample data'
    ]
    return any(indicator in str(text) for indicator in mock_indicators)

def analyze_paper_data(papers):
    """Analyze if papers are real from PubMed"""
    print("\n📚 Analyzing Literature/PubMed Data:")
    print("-" * 70)
    
    if not papers or len(papers) == 0:
        print("   ❌ No papers found")
        return False
    
    real_data_count = 0
    mock_data_count = 0
    
    for i, paper in enumerate(papers[:3]):  # Check first 3 papers
        title = paper.get('title', '')
        pmid = paper.get('pmid', '')
        authors = paper.get('authors', [])
        
        print(f"\n   Paper {i+1}:")
        print(f"   Title: {title[:80]}...")
        print(f"   PMID: {pmid}")
        print(f"   Authors: {len(authors)} authors")
        
        # Check indicators
        if check_if_mock_data(title):
            print("   ⚠️  MOCK DATA DETECTED (generic title)")
            mock_data_count += 1
        elif pmid and (pmid.isdigit() or pmid.startswith('PMID')):
            print("   ✅ REAL DATA (valid PMID)")
            real_data_count += 1
        elif authors and len(authors) > 0:
            print("   ✅ REAL DATA (has authors)")
            real_data_count += 1
        else:
            print("   ⚠️  UNCERTAIN")
    
    if real_data_count > mock_data_count:
        print(f"\n   ✅ VERDICT: REAL PubMed data ({real_data_count} real / {mock_data_count} mock)")
        return True
    else:
        print(f"\n   ⚠️  VERDICT: Likely MOCK data ({real_data_count} real / {mock_data_count} mock)")
        return False

def analyze_trial_data(trials):
    """Analyze if clinical trials are real"""
    print("\n🏥 Analyzing Clinical Trial Data:")
    print("-" * 70)
    
    if not trials or len(trials) == 0:
        print("   ❌ No trials found")
        return False
    
    real_data_count = 0
    mock_data_count = 0
    
    for i, trial in enumerate(trials[:3]):
        nct_id = trial.get('nct_id', '')
        title = trial.get('title', '')
        phase = trial.get('phase', '')
        
        print(f"\n   Trial {i+1}:")
        print(f"   NCT ID: {nct_id}")
        print(f"   Title: {title[:80]}...")
        print(f"   Phase: {phase}")
        
        # Real NCT IDs follow pattern NCT followed by 8 digits
        if re.match(r'^NCT\d{8}$', nct_id):
            print("   ✅ REAL DATA (valid NCT ID format)")
            real_data_count += 1
        elif check_if_mock_data(nct_id) or check_if_mock_data(title):
            print("   ⚠️  MOCK DATA DETECTED")
            mock_data_count += 1
        else:
            print("   ⚠️  UNCERTAIN")
    
    if real_data_count > mock_data_count:
        print(f"\n   ✅ VERDICT: REAL ClinicalTrials.gov data ({real_data_count} real)")
        return True
    else:
        print(f"\n   ⚠️  VERDICT: Likely MOCK data ({mock_data_count} mock)")
        return False

def analyze_safety_data(adverse_events):
    """Analyze safety data"""
    print("\n⚠️  Analyzing Safety/FDA Data:")
    print("-" * 70)
    
    if not adverse_events or len(adverse_events) == 0:
        print("   ℹ️  No adverse events found")
        return None
    
    real_data_count = 0
    
    for i, ae in enumerate(adverse_events[:3]):
        ae_term = ae.get('ae_term', '')
        source = ae.get('source', '')
        
        print(f"\n   Event {i+1}: {ae_term}")
        print(f"   Source: {source}")
        
        if source in ['faers', 'clinicaltrials', 'label', 'fda']:
            print("   ✅ REAL DATA (valid source)")
            real_data_count += 1
        elif check_if_mock_data(ae_term):
            print("   ⚠️  MOCK DATA DETECTED")
    
    if real_data_count > 0:
        print(f"\n   ✅ VERDICT: REAL safety data")
        return True
    else:
        print(f"\n   ⚠️  VERDICT: May be using mock or rule-based data")
        return False

def main():
    print("="*70)
    print("🔍 VERIFICATION TEST - Real Data vs Mock Data")
    print("="*70)
    print("\nSubmitting test query: Aspirin for Cardiovascular Disease")
    print("This will take 30-60 seconds to complete...\n")
    
    # Submit analysis
    payload = {
        "drug_name": "aspirin",
        "indication": "cardiovascular disease"
    }
    
    try:
        print("📤 Submitting to API...")
        response = requests.post(f"{API_URL}/analyze", json=payload, timeout=5)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return
        
        data = response.json()
        job_id = data.get("job_id")
        print(f"✅ Job submitted: {job_id}\n")
        
        # Poll for results
        print("⏳ Waiting for analysis to complete...")
        max_attempts = 90
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(2)
            attempt += 1
            
            status_response = requests.get(f"{API_URL}/jobs/{job_id}", timeout=10)
            
            if status_response.status_code == 200:
                result = status_response.json()
                status = result.get("status")
                
                if attempt % 5 == 0:
                    print(f"   Still processing... ({attempt}/{max_attempts})")
                
                if status == "completed":
                    print("\n✅ Analysis Complete!\n")
                    print("="*70)
                    print("📊 DETAILED ANALYSIS OF RESULTS")
                    print("="*70)
                    
                    data = result.get("data", {})
                    tasks = data.get("tasks", {})
                    
                    results = {}
                    
                    # Check Literature Agent
                    if "literature_agent" in tasks:
                        lit_result = tasks["literature_agent"].get("result", {})
                        papers = lit_result.get("papers", [])
                        results['literature'] = analyze_paper_data(papers)
                    
                    # Check Clinical Agent
                    if "clinical_agent" in tasks:
                        clin_result = tasks["clinical_agent"].get("result", {})
                        trials = clin_result.get("trials", [])
                        results['clinical'] = analyze_trial_data(trials)
                    
                    # Check Safety Agent
                    if "safety_agent" in tasks:
                        safety_result = tasks["safety_agent"].get("result", {})
                        adverse_events = safety_result.get("adverse_events", [])
                        results['safety'] = analyze_safety_data(adverse_events)
                    
                    # Check Molecular Agent
                    if "molecular_agent" in tasks:
                        mol_result = tasks["molecular_agent"].get("result", {})
                        method = mol_result.get("method", "")
                        print(f"\n🧬 Molecular Agent:")
                        print("-" * 70)
                        print(f"   Method: {method}")
                        if "stub" in method.lower():
                            print("   ⚠️  Using rule-based stub (expected)")
                            results['molecular'] = False
                        else:
                            results['molecular'] = True
                    
                    # Final Verdict
                    print("\n" + "="*70)
                    print("🎯 FINAL VERDICT")
                    print("="*70)
                    
                    real_count = sum(1 for v in results.values() if v is True)
                    mock_count = sum(1 for v in results.values() if v is False)
                    
                    print(f"\n✅ Using REAL DATA: {real_count} agents")
                    print(f"⚠️  Using MOCK/STUB: {mock_count} agents")
                    
                    if real_count > mock_count:
                        print("\n🎉 SUCCESS! System IS fetching REAL data from APIs!")
                    elif real_count > 0:
                        print("\n✅ PARTIAL SUCCESS! Some agents using real data, others using mock.")
                    else:
                        print("\n⚠️  WARNING! System appears to be using mostly mock data.")
                    
                    # Save detailed results
                    with open("verification_results.json", "w") as f:
                        json.dump(result, f, indent=2)
                    print("\n💾 Full results saved to: verification_results.json")
                    
                    return
                
                elif status == "failed":
                    print(f"\n❌ Analysis failed: {result.get('error')}")
                    return
        
        print("\n⏰ Timeout - Analysis taking too long")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
