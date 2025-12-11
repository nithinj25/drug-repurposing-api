"""
Literature Agent

Purpose: Locate, parse, extract and summarize scientific literature (PubMed, PMC, preprints)
to produce structured evidence items linking drugs → mechanisms → disease endpoints.

Architecture:
    Source Connectors → Ingestion Layer → Full-text Fetcher → Document Parsing →
    NER & Relation Extraction → Embedding Generation → Indexing → RAG Summarizer → Storage
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import uuid
import logging
import requests
import json
import os
from abc import ABC, abstractmethod

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LangChain imports
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_groq import ChatGroq
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
    from sentence_transformers import SentenceTransformer
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class EvidenceType(str, Enum):
    """Type of scientific evidence"""
    MECHANISM = "mechanism"
    EFFICACY = "efficacy"
    SAFETY = "safety"
    PHARMACOKINETICS = "pharmacokinetics"
    BIOMARKER = "biomarker"
    GENE_EXPRESSION = "gene_expression"
    IN_VITRO = "in_vitro"
    IN_VIVO = "in_vivo"
    CLINICAL_OBSERVATION = "clinical_observation"


class EntityType(str, Enum):
    """Type of named entity"""
    DRUG = "drug"
    PROTEIN = "protein"
    GENE = "gene"
    PATHWAY = "pathway"
    DISEASE = "disease"
    BIOMARKER = "biomarker"
    ORGANISM = "organism"


@dataclass
class Entity:
    """Named entity extracted from literature"""
    entity_id: str
    text: str  # Original mention text
    entity_type: EntityType
    normalized_name: Optional[str] = None  # Canonical name (e.g., HGNC gene symbol)
    confidence: float = 0.8  # 0.0-1.0
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['entity_type'] = self.entity_type.value
        return data


@dataclass
class Relation:
    """Relationship between entities"""
    relation_id: str
    entity1_id: str  # source
    entity2_id: str  # target
    relation_type: str  # e.g., "drug_targets", "causes_disease", "upregulates"
    confidence: float = 0.8
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class QuantitativeResult:
    """Numerical result from study"""
    result_id: str
    metric: str  # e.g., "fold_change", "p_value", "effect_size"
    value: float
    unit: Optional[str] = None
    interpretation: Optional[str] = None  # e.g., "upregulated", "downregulated"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Claim:
    """Claim-level evidence from paper"""
    claim_id: str
    text: str  # 1-3 sentence claim
    evidence_type: EvidenceType
    sentence_index: int  # Position in document
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    quantitative_results: List[QuantitativeResult] = field(default_factory=list)
    confidence_score: float = 0.8
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['evidence_type'] = self.evidence_type.value
        data['entities'] = [e.to_dict() for e in self.entities]
        data['relations'] = [r.to_dict() for r in self.relations]
        data['quantitative_results'] = [qr.to_dict() for qr in self.quantitative_results]
        return data


@dataclass
class PaperMetadata:
    """Scientific paper metadata"""
    paper_id: str  # PMID or custom ID
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    journal: str = ""
    publication_date: Optional[str] = None
    url: str = ""
    source: str = "PubMed"  # PubMed, PMC, bioRxiv, medRxiv
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LiteratureRecord:
    """Complete literature record with extracted evidence"""
    record_id: str
    metadata: PaperMetadata
    full_text: Optional[str] = None  # Extracted from PDF/XML
    sentences: List[str] = field(default_factory=list)  # Segmented sentences
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # Vector embedding of abstract
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        data = {
            'record_id': self.record_id,
            'metadata': self.metadata.to_dict(),
            'full_text': self.full_text[:500] if self.full_text else None,  # Truncate for brevity
            'sentences_count': len(self.sentences),
            'entities': [e.to_dict() for e in self.entities],
            'relations': [r.to_dict() for r in self.relations],
            'claims': [c.to_dict() for c in self.claims],
            'embedding_present': self.embedding is not None,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
        }
        return data


# ============================================================================
# Source Connectors
# ============================================================================

class LiteratureSourceConnector(ABC):
    """Base class for literature source connectors"""
    
    def __init__(self, name: str):
        self.name = name
        self.base_url = ""
        self.drug_name = ""
        self.indication = ""
    
    @abstractmethod
    def search(self, query: str, drug_name: str, limit: int = 10) -> List[Dict]:
        """Search for papers matching query + drug"""
        pass
    
    @abstractmethod
    def fetch_paper(self, paper_id: str) -> Dict:
        """Fetch full paper metadata"""
        pass


class PubMedConnector(LiteratureSourceConnector):
    """Connector to PubMed via Entrez API"""
    
    def __init__(self):
        super().__init__("PubMed")
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.api_key = None  # Set via environment
    
    def search(self, query: str, drug_name: str, limit: int = 10) -> List[Dict]:
        """Search PubMed"""
        logger.info(f"Searching PubMed for {drug_name}: {self.indication}")
        
        # Use stored indication instead of parsing from query
        indication = self.indication or drug_name
        
        # Mock search result (in production: call Entrez API)
        mock_results = [
            {
                "pmid": "36123456",
                "title": f"Molecular mechanisms of {drug_name} in {indication}",
                "abstract": f"This study investigates the molecular mechanisms of {drug_name} efficacy in {indication}. "
                           "We found that the drug targets multiple pathways including protein kinase signaling and mTOR pathway activation. "
                           "These findings suggest potential therapeutic applications.",
                "authors": ["Smith J", "Doe A", "Johnson M"],
                "journal": "Nature Medicine",
                "pub_date": "2023-06-15",
                "doi": "10.1038/nm.2023.12345",
            },
            {
                "pmid": "35987654",
                "title": f"Clinical efficacy of {drug_name} in {indication}: A randomized trial",
                "abstract": f"A randomized controlled trial of {drug_name} in patients with {indication}. "
                           "Primary outcome: 67% of patients showed improvement with mean effect size 1.2. "
                           "Safety profile was favorable with minimal adverse events. Results support further clinical development.",
                "authors": ["Brown K", "Wilson R"],
                "journal": "JAMA",
                "pub_date": "2022-12-01",
                "doi": "10.1001/jama.2022.98765",
            },
        ]
        return mock_results[:limit]
    
    def fetch_paper(self, paper_id: str) -> Dict:
        """Fetch full paper details"""
        logger.info(f"Fetching PubMed paper {paper_id}")
        
        # Mock fetch (in production: call Entrez API)
        return {
            "pmid": paper_id,
            "title": "Study Title",
            "abstract": "Abstract text",
        }


class EuropePMCConnector(LiteratureSourceConnector):
    """Connector to Europe PMC (covers journals and preprints via SRC filters)"""
    
    def __init__(self):
        super().__init__("Europe PMC")
        self.base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        self.timeout_seconds = 8
    
    def _call_api(self, query: str, page_size: int, cursor: str) -> Dict:
        params = {
            "query": query,
            "pageSize": page_size,
            "cursorMark": cursor,
            "resultType": "core",
            "format": "json",
        }
        resp = requests.get(self.base_url, params=params, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()
    
    def search(self, query: str, drug_name: str, limit: int = 10) -> List[Dict]:
        """Search Europe PMC with paging and basic filters"""
        logger.info(f"Searching Europe PMC for {drug_name}: {query}")
        results: List[Dict] = []
        cursor = "*"
        page_size = min(25, max(5, limit))
        try:
            while len(results) < limit and cursor:
                payload = self._call_api(query, page_size, cursor)
                hits = payload.get("resultList", {}).get("result", [])
                for hit in hits:
                    if len(results) >= limit:
                        break
                    # Skip if no abstract or title (low value)
                    if not hit.get("title"):
                        continue
                    results.append({
                        "pmid": hit.get("pmid") or hit.get("id"),
                        "doi": hit.get("doi"),
                        "title": hit.get("title", ""),
                        "abstract": hit.get("abstractText", ""),
                        "authors": [a.strip() for a in hit.get("authorString", "").split(",") if a.strip()],
                        "journal": hit.get("journalTitle", ""),
                        "pub_date": hit.get("firstPublicationDate") or hit.get("pubYear"),
                        "url": hit.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url", ""),
                    })
                # Update cursor; Europe PMC returns None/"" when done
                cursor = payload.get("nextCursorMark")
                if not hits:
                    break
        except Exception as exc:
            logger.warning(f"Europe PMC search failed: {exc}")
        return results
    
    def fetch_paper(self, paper_id: str) -> Dict:
        """Fetch Europe PMC paper (not used; search already returns core fields)"""
        return {}


class BioRxivConnector(LiteratureSourceConnector):
    """Connector to bioRxiv preprints via Europe PMC (SRC:PPR subset)"""
    
    def __init__(self):
        super().__init__("bioRxiv")
        self.epmc = EuropePMCConnector()
    
    def search(self, query: str, drug_name: str, limit: int = 10) -> List[Dict]:
        """Search bioRxiv preprints using Europe PMC SRC:PPR filter"""
        # Europe PMC uses SRC:PPR for preprints (bioRxiv + medRxiv)
        preprint_query = f"({query}) AND SRC:PPR"
        logger.info(f"Searching bioRxiv (via Europe PMC) for {drug_name}: {preprint_query}")
        return self.epmc.search(preprint_query, drug_name, limit)
    
    def fetch_paper(self, paper_id: str) -> Dict:
        return {}


class MedRxivConnector(LiteratureSourceConnector):
    """Connector to medRxiv preprints via Europe PMC (SRC:PPR subset)"""
    
    def __init__(self):
        super().__init__("medRxiv")
        self.epmc = EuropePMCConnector()
    
    def search(self, query: str, drug_name: str, limit: int = 10) -> List[Dict]:
        """Search medRxiv preprints using Europe PMC SRC:PPR filter"""
        preprint_query = f"({query}) AND SRC:PPR"
        logger.info(f"Searching medRxiv (via Europe PMC) for {drug_name}: {preprint_query}")
        return self.epmc.search(preprint_query, drug_name, limit)
    
    def fetch_paper(self, paper_id: str) -> Dict:
        return {}


# ============================================================================
# Ingestion Layer
# ============================================================================

class LiteratureIngestionPipeline:
    """Orchestrates fetching and processing papers from multiple sources"""
    
    def __init__(self):
        self.connectors: Dict[str, LiteratureSourceConnector] = {
            "pubmed": PubMedConnector(),
            "europepmc": EuropePMCConnector(),
            "biorxiv": BioRxivConnector(),
            "medrxiv": MedRxivConnector(),
        }
        self.paper_store: Dict[str, LiteratureRecord] = {}
        self.parser = DocumentParser()
        self.ner_extractor = NERRelationExtractor()
        self.embedder = EmbeddingGenerator()
        logger.info("LiteratureIngestionPipeline initialized with 4 connectors")
    
    def ingest_papers(self, query: str, drug_name: str, indication: str) -> List[LiteratureRecord]:
        """Main ETL pipeline"""
        all_papers: List[LiteratureRecord] = []
        seen_keys = set()
        expanded_query = self._expand_query(query, drug_name)
        
        # Set drug and indication on all connectors for clean mock data
        for connector in self.connectors.values():
            connector.drug_name = drug_name
            connector.indication = indication
        
        for source_name, connector in self.connectors.items():
            try:
                # Pass expanded query for real APIs, connectors will use drug_name + indication for mocks
                raw_results = connector.search(expanded_query, drug_name, limit=8)
                logger.info(f"{source_name}: found {len(raw_results)} papers")
                
                for raw_paper in raw_results:
                    dedup_key = self._dedup_key(raw_paper)
                    if dedup_key in seen_keys:
                        continue
                    seen_keys.add(dedup_key)
                    
                    # Parse paper
                    metadata = self._parse_metadata(raw_paper, source_name)
                    literature_record = LiteratureRecord(
                        record_id=str(uuid.uuid4()),
                        metadata=metadata,
                        full_text=raw_paper.get("abstract", ""),
                    )
                    
                    # Segment sentences
                    literature_record.sentences = self.parser.segment_sentences(
                        literature_record.full_text
                    )
                    
                    # Extract NER and relations
                    literature_record = self.ner_extractor.extract_entities_and_relations(
                        literature_record, drug_name, indication
                    )
                    
                    # Generate embedding
                    literature_record.embedding = self.embedder.embed_text(
                        literature_record.metadata.abstract
                    )
                    
                    # Store
                    self.paper_store[literature_record.record_id] = literature_record
                    all_papers.append(literature_record)
                    logger.info(f"Stored paper {literature_record.record_id}")
            
            except Exception as e:
                logger.warning(f"Error ingesting from {source_name}: {str(e)}")
        
        logger.info(f"Ingest complete: {len(all_papers)} papers processed")
        return all_papers
    
    def _parse_metadata(self, raw_paper: Dict, source: str) -> PaperMetadata:
        """Parse paper metadata"""
        return PaperMetadata(
            paper_id=raw_paper.get("pmid") or str(uuid.uuid4()),
            pmid=raw_paper.get("pmid"),
            doi=raw_paper.get("doi"),
            title=raw_paper.get("title", ""),
            authors=raw_paper.get("authors", []),
            abstract=raw_paper.get("abstract", ""),
            journal=raw_paper.get("journal", ""),
            publication_date=raw_paper.get("pub_date"),
            url=raw_paper.get("url", ""),
            source=source,
        )

    def _expand_query(self, query: str, drug_name: str) -> str:
        """Lightweight query expansion with drug synonyms and common endpoints"""
        synonyms = self.ner_extractor.drug_lexicon.get(drug_name.lower(), [])
        drug_terms = [drug_name] + synonyms
        drug_clause = " OR ".join([f'"{t}"' for t in drug_terms])
        # Bias toward mechanistic and clinical evidence
        evidence_clause = "(mechanism OR pathway OR efficacy OR trial OR safety)"
        return f"({query}) AND ({drug_clause}) AND {evidence_clause}"
    
    def _dedup_key(self, raw_paper: Dict) -> str:
        """Create a stable deduplication key across sources"""
        doi = (raw_paper.get("doi") or "").lower()
        pmid = (raw_paper.get("pmid") or raw_paper.get("id") or "").lower()
        title = (raw_paper.get("title") or "").strip().lower()
        if doi:
            return f"doi:{doi}"
        if pmid:
            return f"pmid:{pmid}"
        if title:
            return f"title:{title[:200]}"
        return str(uuid.uuid4())


# ============================================================================
# Document Parser
# ============================================================================

class DocumentParser:
    """Parses papers and extracts text"""
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences"""
        if not text:
            return []
        
        # Simple sentence segmentation (in production: use NLTK/spaCy)
        sentences = text.replace(".", ".\n").split("\n")
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_full_text(self, pdf_path: str) -> str:
        """Extract text from PDF (mock - in production: use GROBID)"""
        logger.info(f"Extracting text from {pdf_path}")
        return "Full text would be extracted here"


# ============================================================================
# NER & Relation Extraction
# ============================================================================

class NERRelationExtractor:
    """Extracts entities and relations from literature using LangChain + ChatOpenAI"""
    
    def __init__(self):
        self.drug_lexicon = {
            "aspirin": ["acetylsalicylic acid", "asa"],
            "ibuprofen": ["advil", "motrin"],
            "metformin": ["glucophage", "fortamet"],
            "insulin": ["humalog", "lantus"],
        }
        self.pathway_lexicon = {
            "mtor": ["mammalian target of rapamycin", "mtorc1"],
            "akt": ["protein kinase b", "pkb"],
            "mapk": ["mitogen-activated protein kinase"],
            "pi3k": ["phosphoinositide 3-kinase"],
        }
        
        # Initialize LangChain LLM for NER
        self.llm = None
        self.use_llm = False
        
        # Try Groq first if enabled
        if LANGCHAIN_AVAILABLE and os.getenv("USE_GROQ") and os.getenv("GROQ_API_KEY"):
            try:
                self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"))
                self.use_llm = True
                logger.info("NERRelationExtractor: Using Groq (llama-3.1-8b-instant) for NER")
            except Exception as e:
                logger.warning(f"NERRelationExtractor: Failed to initialize Groq: {e}")
        
        # Fallback to OpenAI if Groq not available
        if not self.use_llm and LANGCHAIN_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
                    self.use_llm = True
                    logger.info("NERRelationExtractor: Using ChatOpenAI for NER")
                except Exception as e:
                    logger.warning(f"NERRelationExtractor: Failed to initialize ChatOpenAI: {e}")
        
        if not self.use_llm:
            logger.warning("NERRelationExtractor: No LLM API key, using lexicon fallback")
    
    def extract_entities_and_relations(self, record: LiteratureRecord, 
                                       drug_name: str, indication: str) -> LiteratureRecord:
        """Extract entities and relations from paper using LLM or lexicon fallback"""
        
        if self.use_llm and record.metadata.abstract:
            # Use LLM for entity extraction
            try:
                entities_extracted = self._llm_extract_entities(record.metadata.abstract, drug_name, indication)
                record.entities.extend(entities_extracted)
            except Exception as e:
                logger.warning(f"LLM entity extraction failed: {e}, falling back to lexicon")
                self._lexicon_extract(record, drug_name, indication)
        else:
            # Fallback to lexicon-based extraction
            self._lexicon_extract(record, drug_name, indication)
        
        # Extract relations between entities
        if len(record.entities) >= 2:
            drug_entities = [e for e in record.entities if e.entity_type == EntityType.DRUG]
            for drug_entity in drug_entities:
                for entity in record.entities:
                    if entity.entity_type in [EntityType.PATHWAY, EntityType.PROTEIN, EntityType.GENE]:
                        record.relations.append(Relation(
                            relation_id=str(uuid.uuid4()),
                            entity1_id=drug_entity.entity_id,
                            entity2_id=entity.entity_id,
                            relation_type="targets_pathway",
                            confidence=0.85,
                        ))
        
        # Extract claims
        record.claims = self._extract_claims(record, drug_name, indication)
        
        return record
    
    def _llm_extract_entities(self, text: str, drug_name: str, indication: str) -> List[Entity]:
        """Use ChatOpenAI to extract biomedical entities"""
        prompt = f"""Extract biomedical entities from the following text. Focus on:
- Drugs/compounds
- Proteins/genes
- Pathways
- Diseases

Text: {text[:1000]}

Return entities as JSON list with format: [{{"text": "entity_name", "type": "drug|protein|gene|pathway|disease"}}]
Return ONLY valid JSON, no additional text."""
        
        response = self.llm.invoke(prompt)
        content = response.content.strip()
        
        # Parse JSON response
        try:
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            entities_data = json.loads(content)
            entities = []
            
            for ent_data in entities_data:
                entity_type_map = {
                    "drug": EntityType.DRUG,
                    "protein": EntityType.PROTEIN,
                    "gene": EntityType.GENE,
                    "pathway": EntityType.PATHWAY,
                    "disease": EntityType.DISEASE,
                }
                ent_type = entity_type_map.get(ent_data.get("type", "drug").lower(), EntityType.DRUG)
                
                entities.append(Entity(
                    entity_id=str(uuid.uuid4()),
                    text=ent_data.get("text", ""),
                    entity_type=ent_type,
                    normalized_name=ent_data.get("text", "").lower(),
                    confidence=0.85,
                ))
            
            logger.info(f"LLM extracted {len(entities)} entities")
            return entities
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM entity response: {e}")
            return []
    
    def _lexicon_extract(self, record: LiteratureRecord, drug_name: str, indication: str):
        """Lexicon-based entity extraction fallback"""
        # Extract drugs
        record.entities.append(Entity(
            entity_id=str(uuid.uuid4()),
            text=drug_name,
            entity_type=EntityType.DRUG,
            normalized_name=drug_name.lower(),
            confidence=0.95,
        ))
        
        # Extract disease
        record.entities.append(Entity(
            entity_id=str(uuid.uuid4()),
            text=indication,
            entity_type=EntityType.DISEASE,
            normalized_name=indication.lower(),
            confidence=0.90,
        ))
        
        # Extract pathways from text
        for pathway_name in self.pathway_lexicon.keys():
            if pathway_name in record.metadata.abstract.lower():
                record.entities.append(Entity(
                    entity_id=str(uuid.uuid4()),
                    text=pathway_name.upper(),
                    entity_type=EntityType.PATHWAY,
                    normalized_name=pathway_name,
                    confidence=0.85,
                ))
        
        # Extract relations
        if len(record.entities) >= 2:
            drug_entity = record.entities[0]
            for entity in record.entities[1:]:
                if entity.entity_type == EntityType.PATHWAY:
                    record.relations.append(Relation(
                        relation_id=str(uuid.uuid4()),
                        entity1_id=drug_entity.entity_id,
                        entity2_id=entity.entity_id,
                        relation_type="targets_pathway",
                        confidence=0.85,
                    ))
    
    def _extract_claims(self, record: LiteratureRecord, drug_name: str, 
                       indication: str) -> List[Claim]:
        """Extract claim-level evidence"""
        claims = []
        
        # Mock claim extraction
        claims.append(Claim(
            claim_id=str(uuid.uuid4()),
            text=f"{drug_name} exhibits therapeutic potential in {indication} through multiple molecular mechanisms.",
            evidence_type=EvidenceType.MECHANISM,
            sentence_index=0,
            confidence_score=0.85,
            entities=record.entities[:2],
            relations=record.relations[:1] if record.relations else [],
        ))
        
        # Efficacy claim
        if "efficacy" in record.metadata.abstract.lower() or "improvement" in record.metadata.abstract.lower():
            claims.append(Claim(
                claim_id=str(uuid.uuid4()),
                text=f"Clinical evidence supports {drug_name} efficacy with mean effect size of 1.2 in {indication} patients.",
                evidence_type=EvidenceType.EFFICACY,
                sentence_index=1,
                confidence_score=0.80,
                quantitative_results=[
                    QuantitativeResult(
                        result_id=str(uuid.uuid4()),
                        metric="effect_size",
                        value=1.2,
                        interpretation="moderate efficacy"
                    )
                ]
            ))
        
        return claims


# ============================================================================
# Embedding Generation
# ============================================================================

class EmbeddingGenerator:
    """Generates sentence embeddings for semantic search using sentence-transformers"""
    
    def __init__(self):
        # Use biomedical-optimized model
        try:
            self.model = SentenceTransformer('allenai-specter')
            self.use_model = True
            logger.info("EmbeddingGenerator: Loaded allenai-specter model")
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformers model: {e}, using fallback")
            self.model = None
            self.use_model = False
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not text:
            return []
        
        if self.use_model and self.model:
            try:
                # Generate real embedding
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}, using mock")
        
        # Fallback: Mock embedding (384 dimensions for SPECTER)
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        import random
        random.seed(seed)
        return [random.uniform(-1, 1) for _ in range(384)]


# ============================================================================
# LangChain-Compatible Embedding Wrapper
# ============================================================================

from langchain_core.embeddings import Embeddings

class SentenceTransformerEmbeddings(Embeddings):
    """LangChain Embeddings interface for SentenceTransformer models"""
    
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of texts"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query text"""
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()


# ============================================================================
# Indexing Layer
# ============================================================================

class LiteratureIndexManager:
    """Manages storage and retrieval of papers with FAISS vector store"""
    
    def __init__(self):
        self.relational_store: Dict[str, LiteratureRecord] = {}
        self.vector_store: Optional[FAISS] = None
        self.embedder = EmbeddingGenerator()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        logger.info("LiteratureIndexManager: Initialized with FAISS support")
    
    def store_paper(self, paper: LiteratureRecord):
        """Store paper in relational DB and vector store"""
        self.relational_store[paper.record_id] = paper
        
        # Add to vector store if embedding exists
        if paper.embedding and paper.metadata.abstract:
            try:
                doc = Document(
                    page_content=paper.metadata.abstract,
                    metadata={
                        'record_id': paper.record_id,
                        'pmid': paper.metadata.pmid,
                        'title': paper.metadata.title,
                        'source': paper.metadata.source,
                    }
                )
                
                if self.vector_store is None:
                    # Initialize FAISS with first document using wrapped embedder
                    if self.embedder.use_model:
                        embedding_wrapper = SentenceTransformerEmbeddings(self.embedder.model)
                        self.vector_store = FAISS.from_documents(
                            [doc],
                            embedding=embedding_wrapper
                        )
                    else:
                        logger.warning("Embedder not available, skipping FAISS initialization")
                else:
                    # Add to existing store
                    self.vector_store.add_documents([doc])
                
                logger.info(f"Added paper {paper.record_id} to vector store")
            except Exception as e:
                logger.warning(f"Failed to add paper to vector store: {e}")
        
        logger.info(f"Stored paper {paper.record_id}")
    
    def search_by_keyword(self, keyword: str, drug_name: str) -> List[LiteratureRecord]:
        """Keyword search (in production: Elasticsearch)"""
        results = []
        for paper in self.relational_store.values():
            if keyword.lower() in paper.metadata.title.lower() or \
               keyword.lower() in paper.metadata.abstract.lower():
                if drug_name.lower() in paper.metadata.abstract.lower():
                    results.append(paper)
        return results
    
    def search_by_semantic_similarity(self, query_text: str, top_k: int = 10) -> List[LiteratureRecord]:
        """Semantic search using FAISS vector store"""
        if self.vector_store is None:
            logger.warning("Vector store not initialized, returning empty results")
            return []
        
        try:
            # Use FAISS similarity search
            docs = self.vector_store.similarity_search(query_text, k=top_k)
            
            # Retrieve full records
            results = []
            for doc in docs:
                record_id = doc.metadata.get('record_id')
                if record_id and record_id in self.relational_store:
                    results.append(self.relational_store[record_id])
            
            logger.info(f"Semantic search found {len(results)} papers")
            return results
        
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}, using fallback")
            # Fallback: return all papers
            return list(self.relational_store.values())[:top_k]
    
    def search_by_entities(self, entity_type: EntityType, entity_name: str) -> List[LiteratureRecord]:
        """Search papers containing specific entity"""
        results = []
        for paper in self.relational_store.values():
            for entity in paper.entities:
                if entity.entity_type == entity_type and entity_name.lower() in entity.text.lower():
                    results.append(paper)
                    break
        return results


# ============================================================================
# RAG Summarizer
# ============================================================================

class LiteratureSummarizer:
    """Generates RAG-based evidence summaries using LangChain RetrievalQA"""
    
    def __init__(self):
        # Initialize LangChain LLM
        self.llm = None
        self.use_llm = False
        
        # Try Groq first if enabled
        if LANGCHAIN_AVAILABLE and os.getenv("USE_GROQ") and os.getenv("GROQ_API_KEY"):
            try:
                self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, groq_api_key=os.getenv("GROQ_API_KEY"))
                self.use_llm = True
                logger.info("LiteratureSummarizer: Using Groq (llama-3.1-8b-instant) for summaries")
            except Exception as e:
                logger.warning(f"LiteratureSummarizer: Failed to initialize Groq: {e}")
        
        # Fallback to OpenAI if Groq not available
        if not self.use_llm and LANGCHAIN_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=api_key)
                    self.use_llm = True
                    logger.info("LiteratureSummarizer: Using ChatOpenAI for summaries")
                except Exception as e:
                    logger.warning(f"LiteratureSummarizer: Failed to initialize ChatOpenAI: {e}")
        
        if not self.use_llm:
            logger.warning("LiteratureSummarizer: No LLM API key, using template fallback")
    
    def summarize_evidence(self, papers: List[LiteratureRecord], 
                          drug_name: str, indication: str) -> Dict[str, Any]:
        """Summarize findings across papers using LLM or fallback"""
        
        summary = {
            'drug': drug_name,
            'indication': indication,
            'papers_reviewed': len(papers),
            'total_claims': sum(len(p.claims) for p in papers),
            'mechanisms': [],
            'efficacy_findings': [],
            'safety_concerns': [],
        }
        
        if self.use_llm and papers:
            try:
                llm_summary = self._llm_summarize(papers, drug_name, indication)
                summary['llm_summary'] = llm_summary
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}, using template")
        
        # Aggregate findings (always run for structured data)
        for paper in papers:
            for claim in paper.claims:
                if claim.evidence_type == EvidenceType.MECHANISM:
                    summary['mechanisms'].append({
                        'text': claim.text,
                        'confidence': claim.confidence_score,
                        'source_pmid': paper.metadata.pmid,
                    })
                elif claim.evidence_type == EvidenceType.EFFICACY:
                    summary['efficacy_findings'].append({
                        'text': claim.text,
                        'confidence': claim.confidence_score,
                        'results': [qr.to_dict() for qr in claim.quantitative_results],
                        'source_pmid': paper.metadata.pmid,
                    })
                elif claim.evidence_type == EvidenceType.SAFETY:
                    summary['safety_concerns'].append({
                        'text': claim.text,
                        'confidence': claim.confidence_score,
                        'source_pmid': paper.metadata.pmid,
                    })
        
        return summary
    
    def _llm_summarize(self, papers: List[LiteratureRecord], drug_name: str, indication: str) -> str:
        """Use LLM to generate natural language summary"""
        # Compile evidence text
        evidence_texts = []
        for paper in papers[:5]:  # Limit to top 5 papers to avoid token limits
            text = f"Paper: {paper.metadata.title}\n"
            text += f"Abstract: {paper.metadata.abstract[:500]}\n"
            if paper.claims:
                text += "Claims:\n"
                for claim in paper.claims[:3]:
                    text += f"- {claim.text}\n"
            evidence_texts.append(text)
        
        combined_evidence = "\n\n".join(evidence_texts)
        
        prompt = f"""Summarize the scientific evidence for using {drug_name} to treat {indication}.

Evidence from literature:
{combined_evidence}

Provide a concise summary (3-5 sentences) covering:
1. Mechanisms of action
2. Clinical efficacy
3. Safety considerations

Summary:"""
        
        response = self.llm.invoke(prompt)
        return response.content.strip()


# ============================================================================
# Literature Agent (Main Worker)
# ============================================================================

class LiteratureAgent:
    """Main agent interface for Master Agent"""
    
    def __init__(self):
        self.ingestion_pipeline = LiteratureIngestionPipeline()
        self.index_manager = LiteratureIndexManager()
        self.summarizer = LiteratureSummarizer()
        logger.info("LiteratureAgent initialized")
    
    def query_pubmed(self, query: str, drug_name: str, limit: int = 10) -> List[Dict]:
        """Query PubMed via Entrez API"""
        connector = PubMedConnector()
        return connector.search(query, drug_name, limit)
    
    def extract_claim_sentences(self, paper: LiteratureRecord) -> List[Claim]:
        """Extract claim-level sentences from a paper"""
        return paper.claims
    
    def return_evidence_items(self, papers: List[LiteratureRecord], 
                             drug_name: str, indication: str) -> Dict[str, Any]:
        """Return evidence items with confidence scores"""
        return self.summarizer.summarize_evidence(papers, drug_name, indication)
    
    def run(self, drug_name: str, indication: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main entry point called by Master Agent
        """
        if options is None:
            options = {}
        
        try:
            logger.info(f"Literature Agent: Analyzing {drug_name} for {indication}")
            
            # 1. Ingest papers
            query = f"{drug_name} {indication} mechanism efficacy safety"
            papers = self.ingestion_pipeline.ingest_papers(query, drug_name, indication)
            logger.info(f"Found {len(papers)} papers")
            
            # 2. Index papers
            for paper in papers:
                self.index_manager.store_paper(paper)
            
            # 3. Summarize evidence
            summary = self.summarizer.summarize_evidence(papers, drug_name, indication)
            
            # 4. Compile result
            result = {
                'agent': 'literature_agent',
                'drug': drug_name,
                'indication': indication,
                'papers_found': len(papers),
                'papers': [p.to_dict() for p in papers[:3]],
                'total_claims': sum(len(p.claims) for p in papers),
                'evidence_summary': summary,
                'summary': f"Identified {len(papers)} papers for {drug_name} in {indication}. "
                          f"Extracted {sum(len(p.claims) for p in papers)} claims across mechanism, efficacy, and safety.",
                'status': 'success' if len(papers) > 0 else 'partial',
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            logger.info(f"Literature Agent: Complete. Status={result['status']}")
            return result
        
        except Exception as e:
            logger.error(f"Literature Agent failed: {str(e)}")
            return {
                'agent': 'literature_agent',
                'drug': drug_name,
                'indication': indication,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }


# ============================================================================
# Integration with Master Agent
# ============================================================================

def create_literature_agent_task(master_agent, job_id: str, task_id: str, 
                                 drug_name: str, indication: str, options: Optional[Dict] = None):
    """Worker function for Master Agent integration"""
    agent = LiteratureAgent()
    result = agent.run(drug_name, indication, options)
    
    success = result['status'] != 'failed'
    master_agent.submit_task_result(job_id, task_id, result, success=success)


# ============================================================================
# Demo / Testing
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LITERATURE AGENT - Demo")
    print("="*70 + "\n")
    
    agent = LiteratureAgent()
    
    result = agent.run(
        drug_name="metformin",
        indication="type 2 diabetes",
        options={"include_preprints": True}
    )
    
    print(f"Status: {result['status']}")
    print(f"Papers Found: {result['papers_found']}")
    print(f"Total Claims: {result['total_claims']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nMechanisms Found: {len(result['evidence_summary']['mechanisms'])}")
    print(f"Efficacy Findings: {len(result['evidence_summary']['efficacy_findings'])}")
    print(f"\nFirst Paper:\n{json.dumps(result['papers'][0] if result['papers'] else {}, indent=2)}")