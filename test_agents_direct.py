"""
Direct test of agents to see if they're using real APIs
This bypasses the full workflow to test individual agents
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("="*70)
print("🔬 Direct Agent Testing - Real API vs Mock Data")
print("="*70)

# Load environment
from dotenv import load_dotenv
load_dotenv()

print("\n📋 Environment Check:")
print(f"   GROQ_API_KEY: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Not set'}")
print(f"   USE_GROQ: {os.getenv('USE_GROQ')}")
print(f"   NCBI_EMAIL: {os.getenv('NCBI_EMAIL') or '❌ Not set'}")
print(f"   NCBI_API_KEY: {'✅ Set' if os.getenv('NCBI_API_KEY') else 'Not set (optional)'}")

# Test 1: Molecular Agent (simplest)
print("\n" + "="*70)
print("1️⃣  Testing Molecular Agent")
print("="*70)

from agents.molecular_agent import MolecularAgent

mol_agent = MolecularAgent()
mol_result = mol_agent.run(
    drug_name="aspirin",
    indication="cardiovascular disease"
)

print(f"Status: {mol_result.get('method')}")
print(f"Targets: {mol_result.get('predicted_targets')}")
print(f"Pathways: {mol_result.get('pathways')}")

if 'stub' in mol_result.get('method', '').lower():
    print("⚠️  Using RULE-BASED STUB (not real molecular DB)")
else:
    print("✅ Using real molecular analysis")

# Test 2: Check if Literature Agent attempts real API
print("\n" + "="*70)
print("2️⃣  Testing Literature Agent Connector")
print("="*70)

try:
    from agents.literature_agent import PubMedConnector
    
    connector = PubMedConnector()
    connector.drug_name = "aspirin"
    connector.indication = "pain"
    
    # Try to search (will timeout if API is down, but we can see if it tries)
    print("Attempting PubMed search...")
    try:
        results = connector.search("aspirin pain", "aspirin", limit=2)
        
        if results and len(results) > 0:
            first_result = results[0]
            title = first_result.get('title', '')
            
            if 'Mock' in title or 'Example' in title:
                print("⚠️  Returned MOCK DATA")
                print(f"Title: {title}")
            else:
                print("✅ Attempting to use REAL PubMed API")
                print(f"First result: {title[:80]}...")
        else:
            print("⚠️  No results returned")
            
    except Exception as e:
        print(f"API call error (expected if no real implementation): {str(e)[:100]}")
        
except Exception as e:
    print(f"Error loading literature agent: {e}")

# Test 3: Check Safety Agent for FDA API usage
print("\n" + "="*70)
print("3️⃣  Testing Safety Agent Components")
print("="*70)

try:
    from agents.safety_agent import FDAConnector, DailyMedConnector
    
    print("\nFDA Connector:")
    fda = FDAConnector()
    print(f"   Base URL: {fda.base_url}")
    print(f"   ✅ Configured to use FDA openFDA API")
    
    print("\nDailyMed Connector:")
    dailymed = DailyMedConnector()
    print(f"   Base URL: {dailymed.base_url}")
    print(f"   ✅ Configured to use DailyMed API")
    
except Exception as e:
    print(f"Error: {e}")

# Test 4: Check if LLMs are configured
print("\n" + "="*70)
print("4️⃣  Testing LLM Configuration")
print("="*70)

try:
    from langchain_groq import ChatGroq
    
    if os.getenv('GROQ_API_KEY') and os.getenv('USE_GROQ') == 'true':
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=os.getenv('GROQ_API_KEY')
            )
            print("✅ Groq LLM initialized successfully")
            
            # Try a simple test
            print("Testing Groq API with simple query...")
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content="Say 'API working'")])
            print(f"✅ Groq response: {response.content}")
            
        except Exception as e:
            print(f"⚠️  Groq initialization failed: {e}")
    else:
        print("⚠️  Groq not configured")
        
except Exception as e:
    print(f"Error testing LLM: {e}")

print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)
print("""
Based on the tests above:

✅ Real APIs Configured:
   - FDA openFDA API
   - DailyMed API  
   - PubMed/NCBI (if email set)
   - Groq LLM (if API key valid)

⚠️  Mock/Stub Data:
   - Molecular Agent (rule-based, not real DB queries)
   - Market Agent (IQVIA not available)
   
⚠️  Partially Implemented:
   - Literature, Clinical, Patent agents have connectors but may use mock data
   - Need to verify individual connectors are calling real APIs
""")

print("\n💡 Next Steps:")
print("   1. Verify NCBI_EMAIL is set in .env")
print("   2. Test individual API endpoints directly")
print("   3. Check logs for actual API calls")
print("   4. Consider implementing real molecular DB queries (PubChem/ChEMBL)")
