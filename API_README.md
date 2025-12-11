# Drug Repurposing Assistant API

A REST API wrapper around the multi-agent drug repurposing system. Send drug and indication information, and receive comprehensive analysis with evidence from 6 specialized agents.

## Quick Start

### 1. Start the API Server

```bash
cd c:\Users\Nithin J\OneDrive\Desktop\ey_project\drug-repurposing-assistant
python src/api.py
```

Server starts on `http://localhost:8000`

### 2. Interactive Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
GET /health
```

Returns: API status
```json
{
  "status": "healthy",
  "message": "Drug Repurposing Assistant API is running"
}
```

---

### Analyze Single Drug-Indication Pair
```bash
POST /analyze
```

**Request Body:**
```json
{
  "drug_name": "metformin",
  "indication": "cardiovascular disease",
  "drug_synonyms": ["metformin hydrochloride"],
  "indication_synonyms": ["heart disease", "CVD"],
  "include_patent": true,
  "use_internal_data": false
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "drug_name": "metformin",
    "indication": "cardiovascular disease",
    "status": "completed",
    "tasks": { /* agent results */ },
    "reasoning_result": {
      "composite_score": 0.52,
      "decision_level": "review_required",
      "hypotheses": [ /* ranked hypotheses */ ]
    }
  }
}
```

---

### Batch Analyze Multiple Pairs
```bash
POST /batch
```

**Request Body:**
```json
[
  {
    "drug_name": "metformin",
    "indication": "cardiovascular disease"
  },
  {
    "drug_name": "ibuprofen",
    "indication": "inflammatory bowel disease"
  },
  {
    "drug_name": "aspirin",
    "indication": "diabetes"
  }
]
```

**Response:** Array of analysis results, one per request

---

### Get Job Status
```bash
GET /job/{job_id}
```

**Response:** Same format as `/analyze` response

---

### List All Jobs
```bash
GET /jobs
```

**Response:**
```json
{
  "success": true,
  "total_jobs": 3,
  "jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "drug_name": "metformin",
      "indication": "cardiovascular disease",
      "status": "completed",
      "created_at": "2025-12-10T10:30:00+00:00",
      "tasks_count": 6,
      "human_review_required": true
    }
  ]
}
```

---

### Get Agents Info
```bash
GET /agents
```

**Response:** Information about all 6 specialized agents and reasoning agent

---

## Usage Examples

### Using cURL

```bash
# Single analysis
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "drug_name": "metformin",
    "indication": "cardiovascular disease"
  }'

# Get job status
curl -X GET "http://localhost:8000/job/{job_id}"

# List all jobs
curl -X GET "http://localhost:8000/jobs"
```

### Using Python Requests

```python
import requests
import json

# Single analysis
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "drug_name": "metformin",
        "indication": "cardiovascular disease"
    }
)

result = response.json()
job_id = result["job_id"]

print(f"Job ID: {job_id}")
print(f"Composite Score: {result['data']['reasoning_result']['composite_score']}")
print(f"Decision: {result['data']['reasoning_result']['decision_level']}")

# Get status
status_response = requests.get(f"http://localhost:8000/job/{job_id}")
print(json.dumps(status_response.json(), indent=2))
```

### Using JavaScript/Fetch

```javascript
// Single analysis
const response = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    drug_name: 'metformin',
    indication: 'cardiovascular disease'
  })
});

const result = await response.json();
console.log('Job ID:', result.job_id);
console.log('Composite Score:', result.data.reasoning_result.composite_score);
console.log('Decision:', result.data.reasoning_result.decision_level);
```

---

## Response Structure

### Reasoning Result (Composite Analysis)

```json
{
  "reasoning_result": {
    "composite_score": 0.52,
    "decision_level": "review_required",
    "hypotheses": [
      {
        "rank": 1,
        "hypothesis": "Drug-indication pair shows moderate promise",
        "dimension_scores": {
          "clinical": 0.65,
          "safety": 0.45,
          "patent": 0.52,
          "market": 0.60,
          "molecular": 0.40,
          "regulatory": 0.50
        },
        "supporting_evidence": [ /* evidence items */ ],
        "recommendation": "Recommend further investigation"
      }
    ],
    "constraints": [ /* safety/patent constraints */ ],
    "contradictions": [ /* conflicting evidence */ ],
    "aggregated_evidence": { /* by dimension */ }
  }
}
```

### Agent Results Structure

Each agent (literature, clinical, safety, patent, market, molecular) returns:
- `evidence_items`: List of findings
- `summary`: Text summary
- `evidence_count`: Number of pieces of evidence
- Dimension-specific fields (e.g., safety_score, market_tam, etc.)

---

## Environment Variables

Required `.env` file (see `.env.example`):
```
GROQ_API_KEY=your_key_here
ENTREZ_EMAIL=your_email@example.com
```

---

## Architecture

The API coordinates with:
- **6 Specialized Agents**: Literature, Clinical, Safety, Patent, Market, Molecular
- **Reasoning Agent**: Synthesizes evidence across all dimensions
- **Master Agent**: Orchestrates parallel task execution

All agents run synchronously when you submit a request, ensuring complete results are returned.

---

## Performance

- Single drug-indication analysis: ~5-10 seconds (depending on external API speeds)
- Batch processing: Linear with number of requests
- Parallel agent execution: All 6 agents run concurrently within each analysis

---

## Error Handling

API returns standard HTTP status codes:
- `200 OK`: Success
- `404 Not Found`: Job ID doesn't exist
- `500 Internal Server Error`: Processing error (see response for details)

All errors include detailed error messages for debugging.

---

## Next Steps

1. **Start the server** with `python src/api.py`
2. **Visit the interactive docs** at http://localhost:8000/docs
3. **Try an example request** using the Swagger UI
4. **Integrate with your application** using the endpoint specifications above

For more details, check the Swagger/ReDoc documentation at the server URL.
