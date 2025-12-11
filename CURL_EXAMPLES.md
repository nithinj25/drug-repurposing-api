# Drug Repurposing Assistant API - cURL Examples

## 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

## 2. Analyze Single Drug-Indication Pair

### Basic Request
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "drug_name": "metformin",
    "indication": "cardiovascular disease"
  }'
```

### Full Request with Options
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

### More Examples
```bash
# Aspirin for diabetes
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"drug_name":"aspirin","indication":"diabetes"}'

# Ibuprofen for inflammatory bowel disease
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"drug_name":"ibuprofen","indication":"inflammatory bowel disease"}'

# Simvastatin for Alzheimer's disease
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"drug_name":"simvastatin","indication":"Alzheimer disease"}'
```

## 3. Batch Analyze Multiple Drugs

```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '[
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
  ]'
```

## 4. Get Job Status

Replace `{job_id}` with actual job ID from response:

```bash
curl -X GET "http://localhost:8000/job/{job_id}"
```

Example:
```bash
curl -X GET "http://localhost:8000/job/550e8400-e29b-41d4-a716-446655440000"
```

## 5. List All Jobs

```bash
curl -X GET "http://localhost:8000/jobs"
```

## 6. Get Agent Information

```bash
curl -X GET "http://localhost:8000/agents"
```

## 7. Get API Info

```bash
curl -X GET "http://localhost:8000/"
```

---

## Tips for Windows PowerShell

If using PowerShell, escape quotes differently:

```powershell
$body = @{
    drug_name = "metformin"
    indication = "cardiovascular disease"
} | ConvertTo-Json

curl.exe -X POST "http://localhost:8000/analyze" `
  -H "Content-Type: application/json" `
  -d $body
```

Or use the simpler method:

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/analyze" `
  -Method POST `
  -ContentType "application/json" `
  -Body (@{
    drug_name = "metformin"
    indication = "cardiovascular disease"
  } | ConvertTo-Json)
```

---

## Save Response to File

```bash
# Linux/Mac
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"drug_name":"metformin","indication":"cardiovascular disease"}' \
  > response.json

# Windows PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/analyze" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"drug_name":"metformin","indication":"cardiovascular disease"}' `
  -OutFile response.json
```

---

## Pretty Print JSON Response

```bash
# Using jq (if installed)
curl -X GET "http://localhost:8000/jobs" | jq .

# Using Python
curl -X GET "http://localhost:8000/jobs" | python -m json.tool
```

---

## API Response Structure

All responses follow this format:

```json
{
  "success": true,
  "job_id": "uuid-here",
  "data": {
    "job_id": "uuid-here",
    "drug_name": "metformin",
    "indication": "cardiovascular disease",
    "status": "completed",
    "tasks": { /* agent results */ },
    "reasoning_result": {
      "composite_score": 0.52,
      "decision_level": "review_required",
      "hypotheses": [ /* hypotheses */ ],
      "constraints": [],
      "contradictions": [],
      "aggregated_evidence": {}
    }
  }
}
```

---

## Common Response Codes

- `200 OK` - Success
- `404 Not Found` - Job ID doesn't exist
- `500 Internal Server Error` - Processing error

Check the response body for error details.
