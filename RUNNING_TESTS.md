# Complete Guide: Running API Dry Runs

## Quick Start (3 Steps)

### Step 1: Start the API Server
Open a terminal and run:
```bash
python src/api.py
```

You should see:
```
INFO:     Started server process [12345]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal open!** The API runs in the foreground.

---

### Step 2: Open Another Terminal (in same directory)

Run one of the test scripts:

#### Option A: Simple Test (Recommended for first time)
```bash
python test_simple.py
```

This runs 4 quick tests:
1. Health check (instant)
2. List agents (instant)
3. Analyze drug (5-15 seconds)
4. List jobs (instant)

#### Option B: Interactive Examples
```bash
python run_api_examples.py
```

Menu-driven interface with 6 detailed examples:
1. Health Check
2. Analyze Single Drug
3. List Agents
4. List Jobs
5. Batch Analysis
6. Get Job Result
7. Run All Examples

#### Option C: Run All Dry Runs at Once
```bash
python run_api_examples.py 7
```

---

### Step 3: View Results

Each test outputs:
- ✓ or ✗ status indicators
- API response data
- Formatted results with scores and recommendations

---

## Detailed Walkthrough

### Walking Through test_simple.py

This is the easiest way to get started:

```
[1/4] Testing Health Check...
✓ API is running!
   Status: healthy

[2/4] Getting Available Agents...
✓ Found 6 agents
   - Literature Agent
   - Clinical Agent
   - Safety Agent

[3/4] Analyzing Drug: Metformin → Cardiovascular Disease...
   (This will take 5-15 seconds)
✓ Analysis Complete!
   Job ID: 550e8400-e29b-41d4-a716-446655440000
   Composite Score: 0.52
   Decision: review_required
   
   Dimension Scores:
      - clinical  : 0.65
      - safety    : 0.45
      - patent    : 0.52
      - market    : 0.60
      - molecular : 0.40

[4/4] Listing All Jobs...
✓ Total jobs processed: 1
   1. metformin → cardiovascular disease (completed)

ALL TESTS COMPLETED SUCCESSFULLY! ✓
```

---

## Understanding the Output

### Composite Score (0.0 - 1.0)
- **0.7+**: High potential for repurposing
- **0.3-0.7**: Review required (needs deeper analysis)
- **<0.3**: Low potential

### Dimension Scores
- **Clinical**: Evidence from clinical trials
- **Safety**: Adverse event analysis
- **Patent**: IP/FTO landscape
- **Market**: Market opportunity size
- **Molecular**: Mechanism of action alignment
- **Regulatory**: Regulatory pathway

---

## Running Different Tests

### Test 1: Health Check Only
```bash
python run_api_examples.py 1
```
**Output**: Confirms API is running (instant)

### Test 2: Analyze Single Drug
```bash
python run_api_examples.py 2
```
**Output**: Full analysis with scores (5-15 seconds)

### Test 3: See Available Agents
```bash
python run_api_examples.py 3
```
**Output**: List of 6 agents with descriptions (instant)

### Test 4: See Previous Results
```bash
python run_api_examples.py 4
```
**Output**: All jobs processed during session (instant)

### Test 5: Batch Analyze Multiple Drugs
```bash
python run_api_examples.py 5
```
**Output**: Analysis of 2 drugs at once (10-30 seconds)

### Test 6: Retrieve Specific Job
```bash
python run_api_examples.py 6
```
**Output**: Full results for a particular job (instant)

### Test 7: Run Everything
```bash
python run_api_examples.py 7
```
**Output**: All tests in sequence (30-60 seconds total)

---

## Using with Python Directly

### Example 1: Simple Health Check
```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

Output:
```json
{
  "status": "healthy",
  "message": "Drug Repurposing Assistant API is running"
}
```

### Example 2: Analyze a Drug
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "drug_name": "metformin",
        "indication": "cardiovascular disease"
    }
)

result = response.json()
score = result['data']['reasoning_result']['composite_score']
decision = result['data']['reasoning_result']['decision_level']

print(f"Score: {score}, Decision: {decision}")
```

Output:
```
Score: 0.52, Decision: review_required
```

### Example 3: Batch Analysis
```python
import requests

drugs = [
    {"drug_name": "aspirin", "indication": "diabetes"},
    {"drug_name": "ibuprofen", "indication": "inflammatory bowel disease"},
    {"drug_name": "simvastatin", "indication": "Alzheimer disease"}
]

response = requests.post(
    "http://localhost:8000/batch",
    json=drugs
)

results = response.json()
for job in results['results']:
    print(f"{job['drug_name']}: {job['data']['reasoning_result']['composite_score']:.2f}")
```

Output:
```
aspirin: 0.45
ibuprofen: 0.52
simvastatin: 0.38
```

---

## Using cURL Commands

### Health Check
```bash
curl http://localhost:8000/health
```

### Analyze Single Drug
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"drug_name":"metformin","indication":"cardiovascular disease"}'
```

### List All Jobs
```bash
curl http://localhost:8000/jobs
```

### Get Specific Job
```bash
curl http://localhost:8000/job/YOUR_JOB_ID_HERE
```

---

## Windows: Using Batch Files

### One-Click Test
Double-click: `run_tests.bat`

This will:
1. Check if API is running
2. Run simple tests
3. Show results
4. Keep window open for review

---

## Troubleshooting

### Error: "Could not connect to the remote server"
**Solution**: Make sure the API server is running
```bash
python src/api.py
```
Keep this terminal open while running tests.

### Error: "Connection refused"
**Solution**: Port 8000 might be in use
1. Kill any existing Python processes: `taskkill /im python.exe /f`
2. Start API again: `python src/api.py`

### Error: "Timeout"
**Solution**: First request takes 5-15 seconds. This is normal!
- Subsequent requests are faster
- Make sure you're using timeout=120 in requests

### Error: "500 Internal Server Error"
**Solution**: Check `.env` file
```env
GROQ_API_KEY=your_key_here
ENTREZ_EMAIL=your_email@example.com
```

---

## Performance Expectations

| Operation | Time |
|-----------|------|
| Health check | <100ms |
| List agents | <100ms |
| List jobs | <100ms |
| Single analysis | 5-15 seconds |
| Batch (3 drugs) | 15-45 seconds |
| Cached result | <100ms |

First request takes longest because:
- Models are loaded into memory
- External APIs are queried for first time
- All 6 agents run in parallel

---

## Next Steps After Testing

1. **View Swagger UI**: http://localhost:8000/docs
2. **Try all examples**: Run `python run_api_examples.py 7`
3. **Build your app**: Use `test_simple.py` as reference
4. **Deploy**: Ready for production with docker

---

## File Reference

| File | Purpose | Usage |
|------|---------|-------|
| `test_simple.py` | Quick 4-test demo | `python test_simple.py` |
| `run_api_examples.py` | Interactive menu (6 examples) | `python run_api_examples.py` |
| `test_api_client.py` | Full client library | Import in your code |
| `run_tests.bat` | One-click test (Windows) | Double-click |
| `src/api.py` | API server | `python src/api.py` |

---

## Summary

**To run dry runs:**

1. **Terminal 1** (keep open):
   ```bash
   python src/api.py
   ```

2. **Terminal 2** (new):
   ```bash
   python test_simple.py
   ```

**That's it!** You'll see results from 4 tests in seconds.

For more examples:
```bash
python run_api_examples.py
```

Happy testing! 🚀
