# Drug Repurposing Assistant API - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Start the API Server

#### Option A: Using Python (Recommended)
```bash
cd c:\Users\Nithin J\OneDrive\Desktop\ey_project\drug-repurposing-assistant
python src/api.py
```

#### Option B: Using Batch File (Windows)
Double-click: `start_api_server.bat`

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

### Step 2: Access Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test endpoints directly from the browser!

---

### Step 3: Make Your First Request

#### Using Swagger UI (Easiest)
1. Open http://localhost:8000/docs
2. Expand the "POST /analyze" endpoint
3. Click "Try it out"
4. Modify the example JSON with your drug and indication
5. Click "Execute"

#### Using cURL
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "drug_name": "metformin",
    "indication": "cardiovascular disease"
  }'
```

#### Using PowerShell
```powershell
$body = @{
    drug_name = "metformin"
    indication = "cardiovascular disease"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/analyze" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

#### Using Python
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
print(result)
```

---

## 📊 Understanding the Response

The API returns a JSON response with this structure:

```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "drug_name": "metformin",
    "indication": "cardiovascular disease",
    "status": "completed",
    "human_review_required": true,
    "reasoning_result": {
      "composite_score": 0.52,
      "decision_level": "review_required",
      "hypotheses": [
        {
          "rank": 1,
          "hypothesis": "Metformin shows moderate promise for cardiovascular disease repurposing",
          "dimension_scores": {
            "clinical": 0.65,
            "safety": 0.45,
            "patent": 0.52,
            "market": 0.60,
            "molecular": 0.40,
            "regulatory": 0.50
          },
          "recommendation": "Recommend further investigation"
        }
      ]
    },
    "tasks": {
      "literature": { /* literature agent results */ },
      "clinical": { /* clinical agent results */ },
      "safety": { /* safety agent results */ },
      "patent": { /* patent agent results */ },
      "market": { /* market agent results */ },
      "molecular": { /* molecular agent results */ }
    }
  }
}
```

### Key Fields:

- **composite_score**: 0.0 to 1.0 - Overall repurposing potential
- **decision_level**: 
  - `high_potential` (> 0.7) - Strong candidate
  - `review_required` (0.3-0.7) - Needs deeper analysis
  - `low_potential` (< 0.3) - Limited evidence
- **dimension_scores**: Scores from each specialized agent
  - Clinical: Evidence from clinical trials
  - Safety: Adverse event analysis
  - Patent: IP/FTO landscape
  - Market: Market opportunity
  - Molecular: Mechanism of action
  - Regulatory: Regulatory pathway

---

## 🔄 API Endpoints

### Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/analyze` | POST | Analyze single drug-indication pair |
| `/batch` | POST | Analyze multiple pairs at once |
| `/job/{job_id}` | GET | Retrieve results for a specific job |
| `/jobs` | GET | List all processed jobs |

### Info Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check API health |
| `/agents` | GET | Get available agents info |
| `/docs` | GET | Interactive API documentation |

---

## 💡 Example Use Cases

### 1. Single Drug Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"drug_name":"aspirin","indication":"diabetes"}'
```

### 2. With All Options
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "drug_name": "metformin",
    "indication": "cardiovascular disease",
    "drug_synonyms": ["metformin hydrochloride", "glucophage"],
    "indication_synonyms": ["heart disease", "CVD"],
    "include_patent": true,
    "use_internal_data": false
  }'
```

### 3. Batch Analysis
```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"drug_name":"metformin","indication":"cardiovascular disease"},
    {"drug_name":"ibuprofen","indication":"inflammatory bowel disease"},
    {"drug_name":"aspirin","indication":"diabetes"}
  ]'
```

### 4. Track Job Progress
```bash
curl -X GET "http://localhost:8000/job/550e8400-e29b-41d4-a716-446655440000"
```

### 5. View All Results
```bash
curl -X GET "http://localhost:8000/jobs"
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required: Groq API Key (get from https://console.groq.com)
GROQ_API_KEY=your_api_key_here

# Required: Email for NCBI Entrez (PubMed access)
ENTREZ_EMAIL=your_email@example.com

# Optional: PubMed API Key (for higher rate limits)
PUBMED_API_KEY=your_pubmed_api_key_here

# Optional: API Server Settings
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 🐛 Troubleshooting

### Issue: "Connection refused" error

**Solution**: Make sure the API server is running
```bash
python src/api.py
```

### Issue: 404 error when trying to get job status

**Solution**: Make sure you're using the correct job_id from the analyze response

### Issue: 500 error from API

**Solution**: Check that all environment variables are set properly in `.env`

### Issue: Timeout or slow responses

**Solution**: The first request takes ~10-15 seconds as agents make external API calls. Subsequent requests may be faster depending on caching.

---

## 📈 Performance Tips

1. **Batch Processing**: Use `/batch` endpoint for multiple drugs to save overhead
2. **Caching**: Results are cached - same drug-indication pairs return faster
3. **Job Storage**: All jobs are stored in memory during the session
4. **Parallel Execution**: All 6 agents run in parallel for each analysis

---

## 🔐 Security Notes

- API is currently open to localhost (0.0.0.0:8000)
- For production, add authentication and HTTPS
- Store API keys in environment variables, never hardcode them
- Consider rate limiting for public deployments

---

## 📚 Additional Resources

- **Full API README**: See `API_README.md`
- **cURL Examples**: See `CURL_EXAMPLES.md`
- **Python Client**: See `test_api_client.py` for example usage
- **API Documentation**: Visit http://localhost:8000/docs once server is running

---

## 🚀 Next Steps

1. ✅ Start the API server
2. ✅ Open http://localhost:8000/docs
3. ✅ Try the `/analyze` endpoint with a drug-indication pair
4. ✅ Check the `/jobs` endpoint to see all results
5. ✅ Integrate with your application using the provided client code

---

## 📞 Support

If you encounter issues:
1. Check the API logs in the server terminal
2. Verify environment variables in `.env`
3. Test with the Swagger UI first
4. Check CURL_EXAMPLES.md for syntax help

Happy drug repurposing! 🎉
