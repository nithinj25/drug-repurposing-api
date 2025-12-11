from typing import List, Dict
import requests

def fetch_pubmed_articles(query: str, max_results: int = 20) -> List[Dict]:
    base_url = "https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/"
    params = {
        "query": query,
        "max_results": max_results
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json().get('articles', [])

def extract_claim_sentences(article: Dict) -> List[str]:
    sentences = []
    if 'abstract' in article:
        sentences = article['abstract'].split('. ')
    return sentences

def get_article_evidence(article: Dict) -> Dict:
    evidence = {
        "title": article.get("title"),
        "authors": article.get("authors"),
        "journal": article.get("journal"),
        "confidence_score": calculate_confidence(article)
    }
    return evidence

def calculate_confidence(article: Dict) -> float:
    # Placeholder for confidence score calculation logic
    return 0.75  # Example static confidence score

def process_pubmed_query(query: str) -> List[Dict]:
    articles = fetch_pubmed_articles(query)
    evidence_items = []
    for article in articles:
        claim_sentences = extract_claim_sentences(article)
        for sentence in claim_sentences:
            evidence_item = get_article_evidence(article)
            evidence_items.append(evidence_item)
    return evidence_items