# API Dry Runs - Ready to Test! 🚀

## What You Now Have

3 new test scripts + 1 guide to run the API:

| File | Purpose | How to Run |
|------|---------|-----------|
| `test_simple.py` | **Quick 4-test demo** ← START HERE | `python test_simple.py` |
| `run_api_examples.py` | Interactive menu (6 examples) | `python run_api_examples.py` |
| `run_tests.bat` | One-click Windows test | Double-click |
| `RUNNING_TESTS.md` | Complete guide | Open in editor |

---

## Quickest Way to Test (2 Steps)

### Terminal 1 - Start API Server
```bash
python src/api.py
```

Leave this running!

### Terminal 2 - Run Tests
```bash
python test_simple.py
```

**Done!** You'll see 4 tests run in ~30 seconds:
1. Health check (instant)
2. List agents (instant)
3. Analyze metformin for cardiovascular disease (5-15 sec)
4. List all jobs (instant)

---

## What the Tests Do

### test_simple.py (Easiest - 4 tests)
```
✓ Test 1: Health Check
✓ Test 2: Get Agents
✓ Test 3: Analyze Drug (metformin → cardiovascular disease)
✓ Test 4: List Jobs
```

### run_api_examples.py (Detailed - 6 interactive examples)
```
1. Health Check
2. Analyze Single Drug
3. List Agents
4. List Jobs
5. Batch Analysis (3 drugs)
6. Get Job Result
7. Run All Examples
```

### run_tests.bat (Windows - One-click)
Just double-click and watch tests run!

---

## Sample Output

When you run `python test_simple.py`:

```
======================================================================
  SIMPLE API TEST
======================================================================

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
      - regulatory: 0.50

[4/4] Listing All Jobs...
✓ Total jobs processed: 1
   1. metformin → cardiovascular disease (completed)

======================================================================
  ALL TESTS COMPLETED SUCCESSFULLY! ✓
======================================================================
```

---

## Running Individual Tests

### Just Health Check (instant)
```bash
python run_api_examples.py 1
```

### Just Single Drug Analysis
```bash
python run_api_examples.py 2
```

### Just Batch (3 drugs)
```bash
python run_api_examples.py 5
```

### All 6 Examples in Sequence
```bash
python run_api_examples.py 7
```

---

## Understanding Results

### Composite Score: 0-1 scale
- **0.7+** = Highly promising for repurposing
- **0.3-0.7** = Worth investigating further
- **<0.3** = Limited evidence

### Dimension Breakdown
- **Clinical** (0.65): Evidence from trials/studies
- **Safety** (0.45): Adverse event profile
- **Patent** (0.52): IP/Freedom to operate
- **Market** (0.60): Market size & opportunity
- **Molecular** (0.40): Mechanism alignment
- **Regulatory** (0.50): Regulatory pathway

---

## Python Usage Examples

### In Your Own Code

```python
import requests

# Single analysis
response = requests.post(
    "http://localhost:8000/analyze",
    json={"drug_name": "metformin", "indication": "cardiovascular disease"}
)
result = response.json()
print(result['data']['reasoning_result']['composite_score'])

# Batch analysis
drugs = [
    {"drug_name": "aspirin", "indication": "diabetes"},
    {"drug_name": "ibuprofen", "indication": "inflammatory bowel disease"}
]
response = requests.post("http://localhost:8000/batch", json=drugs)
results = response.json()
print(f"Analyzed {results['total_processed']} drugs")

# List results
response = requests.get("http://localhost:8000/jobs")
jobs = response.json()
print(f"Total jobs: {jobs['total_jobs']}")
```

---

## cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Analyze
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"drug_name":"metformin","indication":"cardiovascular disease"}'

# List jobs
curl http://localhost:8000/jobs

# Get specific job
curl http://localhost:8000/job/YOUR_JOB_ID
```

---

## File Descriptions

### test_simple.py (350 lines)
- Simple, straightforward test
- 4 quick tests
- Good for verifying API works
- Takes ~30 seconds total
- Best for: First-time verification

### run_api_examples.py (500 lines)
- 6 detailed examples
- Interactive menu
- Shows all API capabilities
- Can run individually or all at once
- Best for: Learning all features

### run_tests.bat
- Windows batch wrapper
- Runs test_simple.py
- One-click testing
- Best for: Windows users

### RUNNING_TESTS.md
- Complete testing guide
- Python examples
- cURL examples
- Troubleshooting
- Best for: Reference

---

## Typical Usage Flow

1. **First time?**
   ```bash
   python test_simple.py
   ```
   (Verifies everything works)

2. **Want to explore?**
   ```bash
   python run_api_examples.py
   ```
   (Interactive menu)

3. **Building your app?**
   ```python
   # Use code from run_api_examples.py as reference
   ```

4. **Done testing?**
   ```
   Visit http://localhost:8000/docs
   ```
   (Interactive Swagger UI)

---

## Troubleshooting

**Q: "Could not connect"**
A: Make sure API server is running: `python src/api.py`

**Q: "Request timed out"**
A: Normal for first request (5-15 sec). API loads models on first use.

**Q: "500 error"**
A: Check `.env` file has GROQ_API_KEY and ENTREZ_EMAIL

**Q: "Port 8000 already in use"**
A: Kill Python: `taskkill /im python.exe /f`, then restart

---

## Next Steps

1. ✅ Run: `python test_simple.py`
2. ✅ Try: `python run_api_examples.py 7` (all examples)
3. ✅ Visit: http://localhost:8000/docs (Swagger UI)
4. ✅ Build: Your integration using provided examples

---

## Summary

**You have:**
- ✅ Working API server
- ✅ 3 test scripts
- ✅ 6 example implementations
- ✅ Complete documentation
- ✅ Ready to integrate!

**To test right now:**
```bash
# Terminal 1
python src/api.py

# Terminal 2
python test_simple.py
```

**Done!** 🎉

See `RUNNING_TESTS.md` for more details.
