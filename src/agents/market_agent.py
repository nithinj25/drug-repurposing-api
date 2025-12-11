"""
Market Intelligence Agent

Purpose: Collect, normalize, and synthesize commercial, competitive, and payer/reimbursement
data for drug-indication pairs to produce market snapshots with TAM estimates, CAGR, payer
dynamics, competitor landscape, pricing benchmarks, and go-to-market risk indicators.

Architecture:
    Market Data Sources → Ingestion Pipeline → Entity Normalization → 
    Analytics Engine (TAM/Revenue/Competitor) → Market Report Generation →
    Indexing (Postgres/Elastic) → Master Agent Integration
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta, timezone, UTC
import uuid
import logging
import requests
import json
from abc import ABC, abstractmethod
import os
import re
from statistics import mean, stdev

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_groq import ChatGroq
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
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

class MarketPhase(str, Enum):
    """Market development phase"""
    EMERGING = "emerging"  # Early stage, limited awareness
    GROWTH = "growth"  # Rapid market expansion
    MATURE = "mature"  # Stable market, competition intensive
    DECLINE = "decline"  # Market saturation or replacement


class ReimbursementStatus(str, Enum):
    """Payer reimbursement status"""
    COVERED = "covered"
    RESTRICTED = "restricted"  # Prior auth, quantity limits, etc.
    NOT_COVERED = "not_covered"
    UNDER_REVIEW = "under_review"
    NEGOTIATED = "negotiated"  # Commercial negotiation


class CompetitorThreat(str, Enum):
    """Competitive threat level"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MarketSize:
    """Market size data point"""
    year: int
    market_size_usd: float  # In millions
    units_sold: Optional[int] = None
    average_price: Optional[float] = None
    data_source: str = "IQVIA"
    confidence: float = 0.7  # 0.0-1.0


@dataclass
class TAMEstimate:
    """Total Addressable Market estimate"""
    tam_id: str
    geography: str  # US, EU, Global
    indication: str
    patient_population: int  # Total addressable patients
    average_treatment_cost: float  # USD per patient per year
    penetration_rate: float  # 0.0-1.0
    tam_usd: float  # patient_population * average_treatment_cost * penetration_rate
    cagr_percent: float  # Compound Annual Growth Rate
    forecast_years: int  # Years projected (typically 5-10)
    confidence_level: float  # 0.0-1.0
    methodology: str  # "top-down", "bottom-up", "comparable"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class RevenueScenario:
    """Revenue forecast scenario"""
    scenario_id: str
    scenario_name: str  # Conservative, Median, Aggressive
    indication: str
    launch_year: int
    peak_sales_year: int
    peak_sales_usd: float  # In millions
    market_share_at_peak: float  # 0.0-1.0
    ramp_duration_years: int
    decline_rate_post_peak: float  # Annual decline %
    five_year_revenue: float  # Million USD
    ten_year_revenue: float  # Million USD
    assumptions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CompetitorProgram:
    """Competitive program tracking"""
    program_id: str
    company_name: str
    drug_name: str
    indication: str
    mechanism: str
    development_stage: str  # Phase I/II/III, Approved, etc.
    launch_date: Optional[str] = None
    market_share_estimate: float = 0.0
    launch_price_estimate: Optional[float] = None  # USD per unit/dose
    key_patents: List[str] = field(default_factory=list)
    differentiation_factors: List[str] = field(default_factory=list)
    threat_level: CompetitorThreat = CompetitorThreat.MODERATE
    data_sources: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['threat_level'] = self.threat_level.value
        data['last_updated'] = self.last_updated.isoformat()
        return data


@dataclass
class PricingBenchmark:
    """Pricing data point for competitive analysis"""
    benchmark_id: str
    drug_name: str
    indication: str
    drug_class: str
    pricing_model: str  # Per unit, per month, per year, PMPM
    price_usd: float
    price_currency: str = "USD"
    geography: str = "US"
    payer_type: str = "Commercial"  # Commercial, Medicare, Medicaid
    launch_year: Optional[int] = None
    list_price: Optional[float] = None
    net_price_estimate: Optional[float] = None  # After rebates/discounts
    data_source: str = "ManufacturerReports"
    confidence: float = 0.7


@dataclass
class PayerSignal:
    """Payer and reimbursement intelligence"""
    signal_id: str
    payer_name: str
    indication: str
    reimbursement_status: ReimbursementStatus
    coverage_policy: str  # Detailed policy text
    prior_auth_required: bool = False
    quantity_limits: Optional[str] = None
    formulary_tier: str = "Tier 3"  # Tier 1, 2, 3, 4, or Not Covered
    price_cap_usd: Optional[float] = None
    managed_care_penetration: float = 0.0  # 0.0-1.0
    data_sources: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['reimbursement_status'] = self.reimbursement_status.value
        data['last_updated'] = self.last_updated.isoformat()
        return data


@dataclass
class MarketSnapshot:
    """Complete market intelligence report"""
    snapshot_id: str
    drug_name: str
    indication: str
    analysis_date: datetime = field(default_factory=datetime.utcnow)
    market_phase: MarketPhase = MarketPhase.GROWTH
    geography: str = "US"
    tam_estimate: Optional[TAMEstimate] = None
    revenue_scenarios: List[RevenueScenario] = field(default_factory=list)
    competitors: List[CompetitorProgram] = field(default_factory=list)
    pricing_benchmarks: List[PricingBenchmark] = field(default_factory=list)
    payer_signals: List[PayerSignal] = field(default_factory=list)
    market_size_history: List[MarketSize] = field(default_factory=list)
    go_to_market_risks: List[str] = field(default_factory=list)
    market_summary: str = ""
    key_insights: List[str] = field(default_factory=list)
    data_confidence_score: float = 0.7
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['market_phase'] = self.market_phase.value
        data['analysis_date'] = self.analysis_date.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        data['tam_estimate'] = self.tam_estimate.to_dict() if self.tam_estimate else None
        data['revenue_scenarios'] = [s.to_dict() for s in self.revenue_scenarios]
        data['competitors'] = [c.to_dict() for c in self.competitors]
        data['pricing_benchmarks'] = [p.asdict() for p in self.pricing_benchmarks]
        data['payer_signals'] = [ps.to_dict() for ps in self.payer_signals]
        data['embedding_present'] = self.embedding is not None and len(self.embedding) > 0
        del data['embedding']
        return data


# ============================================================================
# Market Data Source Connectors
# ============================================================================

class MarketDataConnector(ABC):
    """Base class for market data sources"""
    
    def __init__(self, name: str):
        self.name = name
        self.base_url = ""
    
    @abstractmethod
    def fetch_market_data(self, drug_name: str, indication: str) -> Dict:
        """Fetch market data for drug-indication pair"""
        pass
    
    @abstractmethod
    def fetch_competitor_data(self, indication: str) -> List[Dict]:
        """Fetch competitor program data"""
        pass


class IQVIAConnector(MarketDataConnector):
    """Connector to IQVIA market data (mock)"""
    
    def __init__(self):
        super().__init__("IQVIA")
        self.base_url = "https://www.iqvia.com/solutions/market-data"
    
    def fetch_market_data(self, drug_name: str, indication: str) -> Dict:
        """Fetch IQVIA market data"""
        logger.info(f"Fetching IQVIA market data for {drug_name} + {indication}")
        
        # Mock market data
        return {
            "source": "IQVIA",
            "market_sizes": [
                {"year": 2021, "market_size_usd": 1500, "units": 5000000},
                {"year": 2022, "market_size_usd": 1850, "units": 5500000},
                {"year": 2023, "market_size_usd": 2200, "units": 6000000},
            ],
            "growth_rate": 18.5,  # CAGR %
            "patient_population": 3500000,
            "average_cost_per_patient": 650,
        }
    
    def fetch_competitor_data(self, indication: str) -> List[Dict]:
        """Fetch competitor market share and programs"""
        logger.info(f"Fetching IQVIA competitor data for {indication}")
        
        return [
            {
                "company": "Pharma Corp",
                "drug": "DrugX",
                "market_share": 0.35,
                "launch_year": 2019,
                "annual_sales": 850,
            },
            {
                "company": "BioGen Inc",
                "drug": "DrugY",
                "market_share": 0.28,
                "launch_year": 2020,
                "annual_sales": 680,
            },
        ]


class GlobalDataConnector(MarketDataConnector):
    """Connector to GlobalData pharmaceutical intelligence"""
    
    def __init__(self):
        super().__init__("GlobalData")
        self.base_url = "https://www.globaldata.com"
    
    def fetch_market_data(self, drug_name: str, indication: str) -> Dict:
        """Fetch GlobalData market forecasts"""
        logger.info(f"Fetching GlobalData market data for {drug_name}")
        return {}
    
    def fetch_competitor_data(self, indication: str) -> List[Dict]:
        """Fetch pipeline and competitor programs"""
        return []


class EvaluatePharmaConnector(MarketDataConnector):
    """Connector to EvaluatePharma industry data"""
    
    def __init__(self):
        super().__init__("EvaluatePharma")
        self.base_url = "https://www.evaluate.com"
    
    def fetch_market_data(self, drug_name: str, indication: str) -> Dict:
        """Fetch EvaluatePharma market data"""
        return {}
    
    def fetch_competitor_data(self, indication: str) -> List[Dict]:
        """Fetch competitor programs"""
        return []


# ============================================================================
# Entity Normalization with LangChain
# ============================================================================

class EntityNormalizer:
    """Normalize and canonicalize company names, drug names, and indications using LLM"""
    
    def __init__(self):
        # Initialize LangChain LLM for entity normalization
        self.use_llm = False
        self.llm = None
        
        # Try Groq first if enabled
        if LANGCHAIN_AVAILABLE and os.getenv("USE_GROQ") and os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
                self.use_llm = True
                logger.info("EntityNormalizer: Using Groq (llama-3.1-8b-instant) for entity normalization")
            except Exception as e:
                logger.warning(f"EntityNormalizer: Failed to initialize Groq: {e}")
        
        # Fallback to OpenAI if Groq not available
        if not self.use_llm and LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                self.use_llm = True
                logger.info("EntityNormalizer: Using ChatOpenAI for entity normalization")
            except Exception as e:
                logger.warning(f"EntityNormalizer: Failed to initialize ChatOpenAI: {e}")
        
        if not self.use_llm:
            logger.warning("EntityNormalizer: No LLM API key, using rule-based fallback")
        
        # Canonical reference data
        self.company_mappings = {
            "pharma corp": "Pharma Corporation",
            "pharmatech": "Pharma Corporation",
            "biogen": "BioGen Inc",
            "biogentech": "BioGen Inc",
        }
        
        self.drug_mappings = {
            "aspirin": {"brand": "Bayer Aspirin", "inn": "acetylsalicylic acid", "cas": "50-78-2"},
            "metformin": {"brand": "Glucophage", "inn": "metformin hydrochloride", "cas": "1115-70-4"},
            "ibuprofen": {"brand": "Advil", "inn": "ibuprofen", "cas": "15687-27-1"},
        }
        
        self.indication_mappings = {
            "diabetes": "Type 2 Diabetes Mellitus",
            "t2dm": "Type 2 Diabetes Mellitus",
            "cardiovascular": "Cardiovascular Disease",
            "cvd": "Cardiovascular Disease",
            "pain": "Chronic Pain Management",
            "arthritis": "Rheumatoid Arthritis",
            "ra": "Rheumatoid Arthritis",
        }
    
    def normalize_company(self, company_name: str) -> str:
        """Normalize company name"""
        if self.use_llm:
            return self._llm_normalize_company(company_name)
        else:
            return self._rule_normalize_company(company_name)
    
    def normalize_drug(self, drug_name: str) -> Dict[str, str]:
        """Normalize drug name and return brand/INN/CAS"""
        if self.use_llm:
            return self._llm_normalize_drug(drug_name)
        else:
            return self._rule_normalize_drug(drug_name)
    
    def normalize_indication(self, indication: str) -> str:
        """Normalize indication/disease name"""
        if self.use_llm:
            return self._llm_normalize_indication(indication)
        else:
            return self._rule_normalize_indication(indication)
    
    def _llm_normalize_company(self, company_name: str) -> str:
        """Use LLM to normalize company name"""
        prompt = f"""Normalize this company name to its canonical form.
        
Company: {company_name}

Return ONLY the canonical company name, no additional text."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LLM company normalization failed: {e}")
            return self._rule_normalize_company(company_name)
    
    def _rule_normalize_company(self, company_name: str) -> str:
        """Rule-based company name normalization"""
        company_lower = company_name.lower()
        for variant, canonical in self.company_mappings.items():
            if variant in company_lower:
                return canonical
        return company_name.title()
    
    def _llm_normalize_drug(self, drug_name: str) -> Dict[str, str]:
        """Use LLM to normalize drug"""
        prompt = f"""Normalize this drug name and provide brand name, INN, and CAS number if available.

Drug: {drug_name}

Return in format:
BRAND: <brand name>
INN: <international nonproprietary name>
CAS: <CAS number or "N/A">"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            brand_match = re.search(r'BRAND:\s*(.+)', content)
            inn_match = re.search(r'INN:\s*(.+)', content)
            cas_match = re.search(r'CAS:\s*(.+)', content)
            
            return {
                'brand': brand_match.group(1).strip() if brand_match else drug_name,
                'inn': inn_match.group(1).strip() if inn_match else drug_name,
                'cas': cas_match.group(1).strip() if cas_match else "N/A",
            }
        except Exception as e:
            logger.warning(f"LLM drug normalization failed: {e}")
            return self._rule_normalize_drug(drug_name)
    
    def _rule_normalize_drug(self, drug_name: str) -> Dict[str, str]:
        """Rule-based drug normalization"""
        drug_lower = drug_name.lower()
        if drug_lower in self.drug_mappings:
            return self.drug_mappings[drug_lower]
        return {
            'brand': drug_name,
            'inn': drug_name.lower(),
            'cas': "N/A",
        }
    
    def _llm_normalize_indication(self, indication: str) -> str:
        """Use LLM to normalize indication"""
        prompt = f"""Normalize this disease/indication to MeSH or standard terminology.

Indication: {indication}

Return ONLY the canonical indication name, no additional text."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LLM indication normalization failed: {e}")
            return self._rule_normalize_indication(indication)
    
    def _rule_normalize_indication(self, indication: str) -> str:
        """Rule-based indication normalization"""
        indication_lower = indication.lower()
        for variant, canonical in self.indication_mappings.items():
            if variant in indication_lower:
                return canonical
        return indication.title()


# ============================================================================
# Analytics Engine
# ============================================================================

class MarketAnalyticsEngine:
    """Calculate TAM, revenue scenarios, competitor analysis"""
    
    def __init__(self):
        # Initialize LangChain LLM for market insights
        self.use_llm = False
        self.llm = None
        
        # Try Groq first if enabled
        if LANGCHAIN_AVAILABLE and os.getenv("USE_GROQ") and os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
                self.use_llm = True
                logger.info("MarketAnalyticsEngine: Using Groq (llama-3.1-8b-instant) for market insights")
            except Exception as e:
                logger.warning(f"MarketAnalyticsEngine: Failed to initialize Groq: {e}")
        
        # Fallback to OpenAI if Groq not available
        if not self.use_llm and LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
                self.use_llm = True
                logger.info("MarketAnalyticsEngine: Using ChatOpenAI for market insights")
            except Exception as e:
                logger.warning(f"MarketAnalyticsEngine: Failed to initialize ChatOpenAI: {e}")
        
        if not self.use_llm:
            logger.warning("MarketAnalyticsEngine: No LLM API key, using calculation fallback")
    
    def estimate_tam(self, indication: str, patient_population: int,
                    average_cost: float, penetration_rate: float = 0.5) -> TAMEstimate:
        """Estimate Total Addressable Market"""
        
        tam_usd = patient_population * average_cost * penetration_rate
        
        # Estimate CAGR based on therapeutic area
        cagr = self._estimate_cagr(indication)
        
        return TAMEstimate(
            tam_id=str(uuid.uuid4()),
            geography="US",
            indication=indication,
            patient_population=patient_population,
            average_treatment_cost=average_cost,
            penetration_rate=penetration_rate,
            tam_usd=tam_usd,
            cagr_percent=cagr,
            forecast_years=10,
            confidence_level=0.75,
            methodology="bottom-up",
        )
    
    def _estimate_cagr(self, indication: str) -> float:
        """Estimate CAGR based on indication and market trends"""
        indication_lower = indication.lower()
        
        # Typical CAGR ranges by therapeutic area
        if "diabetes" in indication_lower:
            return 7.5
        elif "cardiovascular" in indication_lower:
            return 6.2
        elif "oncology" in indication_lower:
            return 12.3
        elif "immunology" in indication_lower:
            return 9.8
        elif "pain" in indication_lower:
            return 5.1
        else:
            return 8.0
    
    def generate_revenue_scenarios(self, tam_estimate: TAMEstimate,
                                  market_share_estimates: Dict[str, float]) -> List[RevenueScenario]:
        """Generate conservative/median/aggressive revenue scenarios"""
        
        scenarios = []
        
        # Conservative scenario
        scenarios.append(RevenueScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name="Conservative",
            indication=tam_estimate.indication,
            launch_year=2026,
            peak_sales_year=2031,
            peak_sales_usd=tam_estimate.tam_usd * 0.08,  # 8% market share
            market_share_at_peak=0.08,
            ramp_duration_years=5,
            decline_rate_post_peak=0.02,
            five_year_revenue=tam_estimate.tam_usd * 0.04 * 5,
            ten_year_revenue=tam_estimate.tam_usd * 0.06 * 10,
            assumptions=[
                "Delayed market adoption",
                "Limited formulary coverage",
                "Pricing pressure from competition",
            ],
        ))
        
        # Median scenario
        scenarios.append(RevenueScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name="Median",
            indication=tam_estimate.indication,
            launch_year=2025,
            peak_sales_year=2030,
            peak_sales_usd=tam_estimate.tam_usd * 0.15,  # 15% market share
            market_share_at_peak=0.15,
            ramp_duration_years=5,
            decline_rate_post_peak=0.01,
            five_year_revenue=tam_estimate.tam_usd * 0.08 * 5,
            ten_year_revenue=tam_estimate.tam_usd * 0.12 * 10,
            assumptions=[
                "Moderate market adoption",
                "Broad payer coverage",
                "Effective differentiation vs competitors",
            ],
        ))
        
        # Aggressive scenario
        scenarios.append(RevenueScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name="Aggressive",
            indication=tam_estimate.indication,
            launch_year=2024,
            peak_sales_year=2029,
            peak_sales_usd=tam_estimate.tam_usd * 0.25,  # 25% market share
            market_share_at_peak=0.25,
            ramp_duration_years=4,
            decline_rate_post_peak=0.005,
            five_year_revenue=tam_estimate.tam_usd * 0.15 * 5,
            ten_year_revenue=tam_estimate.tam_usd * 0.20 * 10,
            assumptions=[
                "Rapid market adoption",
                "Premium pricing maintained",
                "Market expansion through new indications",
            ],
        ))
        
        return scenarios
    
    def assess_competitor_threat(self, competitors: List[CompetitorProgram],
                                drug_launch_year: int = 2025) -> Tuple[List[CompetitorProgram], str]:
        """Assess competitive threat levels based on development stage and timing"""
        
        threat_summary = ""
        
        for competitor in competitors:
            # Parse development stage
            stage = competitor.development_stage.lower()
            
            # Determine threat level
            if "approved" in stage or "marketed" in stage:
                threat_level = CompetitorThreat.CRITICAL
            elif "phase 3" in stage or "registration" in stage:
                threat_level = CompetitorThreat.HIGH
            elif "phase 2" in stage:
                threat_level = CompetitorThreat.MODERATE
            else:
                threat_level = CompetitorThreat.LOW
            
            # Adjust for launch timing
            if competitor.launch_date:
                try:
                    launch = datetime.fromisoformat(competitor.launch_date)
                    time_to_market = (launch - datetime.now(UTC)).days / 365
                    if time_to_market < 1 and threat_level == CompetitorThreat.MODERATE:
                        threat_level = CompetitorThreat.HIGH
                except:
                    pass
            
            competitor.threat_level = threat_level
        
        # Generate summary
        critical_count = len([c for c in competitors if c.threat_level == CompetitorThreat.CRITICAL])
        high_count = len([c for c in competitors if c.threat_level == CompetitorThreat.HIGH])
        
        if critical_count > 2:
            threat_summary = f"HIGHLY COMPETITIVE: {critical_count} approved competitors + {high_count} in Phase III"
        elif critical_count > 0:
            threat_summary = f"COMPETITIVE: {critical_count} approved competitors, differentiation critical"
        else:
            threat_summary = "MODERATE COMPETITION: Pre-commercial competitors provide window for market entry"
        
        return competitors, threat_summary
    
    def generate_market_insights(self, snapshot: "MarketSnapshot") -> List[str]:
        """Generate business insights using LLM"""
        
        insights = []
        
        if self.use_llm:
            insights = self._llm_generate_insights(snapshot)
        else:
            insights = self._rule_generate_insights(snapshot)
        
        return insights
    
    def _llm_generate_insights(self, snapshot: "MarketSnapshot") -> List[str]:
        """Use LLM to generate market insights"""
        
        # Compile market context
        context = f"""Market Analysis Context:
Drug: {snapshot.drug_name}
Indication: {snapshot.indication}
TAM: ${snapshot.tam_estimate.tam_usd:.0f}M
CAGR: {snapshot.tam_estimate.cagr_percent}%
Competitors: {len(snapshot.competitors)}
Market Phase: {snapshot.market_phase.value}"""
        
        prompt = f"""{context}

Generate 3-4 concise business insights (1-2 sentences each) for go-to-market strategy.
Focus on: market opportunity, competitive landscape, payer dynamics, pricing strategy."""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Parse insights (split by newlines or bullet points)
            insights = [s.strip() for s in content.split('\n') if s.strip() and len(s) > 20]
            return insights[:4]
        
        except Exception as e:
            logger.warning(f"LLM insight generation failed: {e}")
            return self._rule_generate_insights(snapshot)
    
    def _rule_generate_insights(self, snapshot: "MarketSnapshot") -> List[str]:
        """Rule-based insight generation"""
        
        insights = []
        
        # TAM insight
        if snapshot.tam_estimate:
            insights.append(
                f"TAM of ${snapshot.tam_estimate.tam_usd:.0f}M with {snapshot.tam_estimate.cagr_percent}% "
                f"CAGR represents significant commercial opportunity in {snapshot.indication}."
            )
        
        # Competitive insight
        critical_competitors = [c for c in snapshot.competitors if c.threat_level == CompetitorThreat.CRITICAL]
        if critical_competitors:
            insights.append(
                f"{len(critical_competitors)} approved competitors already in market - success requires "
                f"clear differentiation on efficacy, safety, or dosing convenience."
            )
        else:
            insights.append(
                f"Limited approved competition provides window for market entry; focus on rapid Phase III "
                f"completion and payer engagement."
            )
        
        # Payer insight
        if snapshot.payer_signals:
            restricted = [p for p in snapshot.payer_signals if p.reimbursement_status == ReimbursementStatus.RESTRICTED]
            if restricted:
                insights.append(
                    f"{len(restricted)} major payers have restrictions (prior auth/quantity limits); "
                    f"pricing strategy must align with existing reimbursement precedents."
                )
        
        return insights


# ============================================================================
# Market Data Ingestion Pipeline
# ============================================================================

class MarketIngestionPipeline:
    """Orchestrate market data fetching, normalization, and analysis"""
    
    def __init__(self):
        self.connectors: Dict[str, MarketDataConnector] = {
            "iqvia": IQVIAConnector(),
            "globaldata": GlobalDataConnector(),
            "evaluate": EvaluatePharmaConnector(),
        }
        self.normalizer = EntityNormalizer()
        self.analytics = MarketAnalyticsEngine()
        self.market_store: Dict[str, MarketSnapshot] = {}
        logger.info("MarketIngestionPipeline initialized with 3 data connectors")
    
    def ingest_market_data(self, drug_name: str, indication: str,
                          geography: str = "US") -> MarketSnapshot:
        """Main market data ingestion and analysis pipeline"""
        
        logger.info(f"Market Pipeline: Analyzing {drug_name} for {indication}")
        
        # Normalize inputs
        normalized_drug = self.normalizer.normalize_drug(drug_name)
        normalized_indication = self.normalizer.normalize_indication(indication)
        
        logger.info(f"Normalized: {normalized_drug['brand']} | {normalized_indication}")
        
        # Fetch market data from connectors
        market_data = {}
        competitor_data = []
        
        for source_name, connector in self.connectors.items():
            try:
                data = connector.fetch_market_data(drug_name, indication)
                if data:
                    market_data[source_name] = data
                
                competitors = connector.fetch_competitor_data(indication)
                if competitors:
                    competitor_data.extend(competitors)
            
            except Exception as e:
                logger.warning(f"Error fetching from {source_name}: {e}")
        
        # Create snapshot
        snapshot = MarketSnapshot(
            snapshot_id=str(uuid.uuid4()),
            drug_name=normalized_drug['brand'],
            indication=normalized_indication,
            geography=geography,
            market_phase=self._determine_market_phase(indication),
        )
        
        # Estimate TAM
        if market_data:
            iqvia = market_data.get('iqvia', {})
            patient_pop = iqvia.get('patient_population', 2000000)
            avg_cost = iqvia.get('average_cost_per_patient', 500)
            snapshot.tam_estimate = self.analytics.estimate_tam(
                indication, patient_pop, avg_cost
            )
        
        # Generate revenue scenarios
        if snapshot.tam_estimate:
            snapshot.revenue_scenarios = self.analytics.generate_revenue_scenarios(
                snapshot.tam_estimate, {}
            )
        
        # Process competitors
        competitor_programs = self._normalize_competitors(competitor_data)
        competitor_programs, threat_summary = self.analytics.assess_competitor_threat(
            competitor_programs
        )
        snapshot.competitors = competitor_programs
        
        # Mock payer signals
        snapshot.payer_signals = self._generate_payer_signals(indication)
        
        # Generate insights
        snapshot.key_insights = self.analytics.generate_market_insights(snapshot)
        
        # Generate summary
        snapshot.market_summary = self._generate_market_summary(snapshot)
        
        # Compute confidence
        snapshot.data_confidence_score = 0.7 + (0.1 * len(self.connectors))
        
        # Store
        self.market_store[snapshot.snapshot_id] = snapshot
        
        logger.info(f"Market analysis complete: TAM=${snapshot.tam_estimate.tam_usd:.0f}M")
        return snapshot
    
    def _determine_market_phase(self, indication: str) -> MarketPhase:
        """Determine market development phase"""
        indication_lower = indication.lower()
        
        if "orphan" in indication_lower or "rare" in indication_lower:
            return MarketPhase.EMERGING
        elif "oncology" in indication_lower or "advanced" in indication_lower:
            return MarketPhase.MATURE
        else:
            return MarketPhase.GROWTH
    
    def _normalize_competitors(self, competitor_data: List[Dict]) -> List[CompetitorProgram]:
        """Normalize competitor data"""
        
        programs = []
        for comp in competitor_data:
            normalized_company = self.normalizer.normalize_company(comp.get('company', ''))
            
            program = CompetitorProgram(
                program_id=str(uuid.uuid4()),
                company_name=normalized_company,
                drug_name=comp.get('drug', ''),
                indication="",  # Would be normalized
                mechanism="Not specified",
                development_stage=comp.get('stage', 'Marketed'),
                launch_date=f"{comp.get('launch_year', 2024)}-01-01",
                market_share_estimate=comp.get('market_share', 0.0),
                data_sources=["IQVIA"],
            )
            programs.append(program)
        
        return programs
    
    def _generate_payer_signals(self, indication: str) -> List[PayerSignal]:
        """Generate mock payer signals"""
        
        signals = [
            PayerSignal(
                signal_id=str(uuid.uuid4()),
                payer_name="UnitedHealth",
                indication=indication,
                reimbursement_status=ReimbursementStatus.COVERED,
                coverage_policy="Coverage with no quantity limits for FDA-approved indication",
                formulary_tier="Tier 2",
                managed_care_penetration=0.28,
            ),
            PayerSignal(
                signal_id=str(uuid.uuid4()),
                payer_name="Aetna",
                indication=indication,
                reimbursement_status=ReimbursementStatus.RESTRICTED,
                coverage_policy="Prior authorization required. Limited to 1 fill per month.",
                prior_auth_required=True,
                quantity_limits="1 fill per month",
                formulary_tier="Tier 3",
                managed_care_penetration=0.18,
            ),
        ]
        
        return signals
    
    def _generate_market_summary(self, snapshot: MarketSnapshot) -> str:
        """Generate human-readable market summary"""
        
        summary = f"The {snapshot.indication} market in {snapshot.geography} represents a TAM of "
        
        if snapshot.tam_estimate:
            summary += f"${snapshot.tam_estimate.tam_usd:.0f}M growing at {snapshot.tam_estimate.cagr_percent}% CAGR. "
        
        summary += f"Market is characterized as {snapshot.market_phase.value}. "
        
        if snapshot.competitors:
            threat_summary = "Limited to moderate competition" if len(snapshot.competitors) < 3 else "Competitive landscape"
            summary += f"{threat_summary} with {len(snapshot.competitors)} key competitors. "
        
        if snapshot.revenue_scenarios:
            median = next((s for s in snapshot.revenue_scenarios if s.scenario_name == "Median"), None)
            if median:
                summary += f"Median peak sales potential of ${median.peak_sales_usd:.0f}M by {median.peak_sales_year}."
        
        return summary


# ============================================================================
# Embedding Generator with LangChain
# ============================================================================

class MarketEmbeddingGenerator:
    """Generate embeddings for market data using SentenceTransformer"""
    
    def __init__(self):
        self.use_model = False
        self.model = None
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.model = SentenceTransformer('allenai-specter')
                self.use_model = True
                logger.info("MarketEmbeddingGenerator: Loaded allenai-specter model")
            except Exception as e:
                logger.warning(f"MarketEmbeddingGenerator: Failed to load model: {e}")
        
        if not self.use_model:
            logger.warning("MarketEmbeddingGenerator: Using mock embeddings")
    
    def embed_market_snapshot(self, snapshot: MarketSnapshot) -> List[float]:
        """Generate embedding for market snapshot"""
        
        # Combine relevant text for embedding
        text = f"{snapshot.drug_name} {snapshot.indication}. {snapshot.market_summary}"
        
        return self.embed_text(text)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for any text"""
        if not text:
            return []
        
        if self.use_model and self.model:
            try:
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}, using mock")
        
        # Fallback: Mock embedding
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        import random
        random.seed(seed)
        return [random.uniform(-1, 1) for _ in range(384)]


# ============================================================================
# Market Index Manager with FAISS
# ============================================================================

class MarketIndexManager:
    """Manage market data storage and FAISS-based semantic search"""
    
    def __init__(self):
        self.relational_store: Dict[str, MarketSnapshot] = {}
        self.vector_store = None
        self.embedder = MarketEmbeddingGenerator()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        ) if LANGCHAIN_AVAILABLE else None
        logger.info("MarketIndexManager: Initialized with FAISS support")
    
    def store_market_snapshot(self, snapshot: MarketSnapshot):
        """Store market snapshot in relational and vector stores"""
        self.relational_store[snapshot.snapshot_id] = snapshot
        
        # Generate and store embedding
        snapshot.embedding = self.embedder.embed_market_snapshot(snapshot)
        
        # Add to vector store
        if LANGCHAIN_AVAILABLE and self.text_splitter and snapshot.embedding:
            try:
                doc = Document(
                    page_content=snapshot.market_summary,
                    metadata={
                        'snapshot_id': snapshot.snapshot_id,
                        'drug': snapshot.drug_name,
                        'indication': snapshot.indication,
                        'market_phase': snapshot.market_phase.value,
                        'geography': snapshot.geography,
                    }
                )
                
                if self.vector_store is None:
                    # Initialize FAISS
                    if self.embedder.use_model:
                        from langchain_community.embeddings.fake import FakeEmbeddings
                        fake_embedder = FakeEmbeddings(size=384)
                        self.vector_store = FAISS.from_documents([doc], embedding=fake_embedder)
                    else:
                        logger.warning("Embedder not available, skipping FAISS")
                else:
                    self.vector_store.add_documents([doc])
                
                logger.info(f"Added market snapshot {snapshot.snapshot_id} to vector store")
            except Exception as e:
                logger.warning(f"Failed to add snapshot to vector store: {e}")
        
        logger.info(f"Stored market snapshot {snapshot.snapshot_id}")


# ============================================================================
# Market Intelligence Agent (Main Worker)
# ============================================================================

class MarketAgent:
    """Main agent interface for Master Agent"""
    
    def __init__(self):
        self.ingestion_pipeline = MarketIngestionPipeline()
        self.index_manager = MarketIndexManager()
        logger.info("MarketAgent initialized")
    
    def run(self, drug_name: str, indication: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main entry point called by Master Agent.
        
        Returns:
            {
                'agent': 'market_agent',
                'drug': drug_name,
                'indication': indication,
                'market_snapshot': Dict,
                'tam_estimate': Dict,
                'revenue_scenarios': List[Dict],
                'competitors': List[Dict],
                'payer_signals': List[Dict],
                'key_insights': List[str],
                'go_to_market_risks': List[str],
                'status': 'success' | 'partial' | 'failed',
            }
        """
        if options is None:
            options = {}
        
        try:
            logger.info(f"Market Agent: Analyzing {drug_name} for {indication}")
            
            # 1. Ingest and analyze market data
            snapshot = self.ingestion_pipeline.ingest_market_data(
                drug_name, indication,
                geography=options.get('geography', 'US')
            )
            
            # 2. Index snapshot
            self.index_manager.store_market_snapshot(snapshot)
            
            # 3. Identify go-to-market risks
            gtm_risks = self._identify_gtm_risks(snapshot)
            snapshot.go_to_market_risks = gtm_risks
            
            # 4. Compile result
            result = {
                'agent': 'market_agent',
                'drug': snapshot.drug_name,
                'indication': snapshot.indication,
                'market_snapshot': snapshot.to_dict(),
                'tam_estimate': snapshot.tam_estimate.to_dict() if snapshot.tam_estimate else None,
                'revenue_scenarios': [s.to_dict() for s in snapshot.revenue_scenarios],
                'competitors': [c.to_dict() for c in snapshot.competitors],
                'payer_signals': [p.to_dict() for p in snapshot.payer_signals],
                'key_insights': snapshot.key_insights,
                'go_to_market_risks': gtm_risks,
                'market_phase': snapshot.market_phase.value,
                'data_confidence': snapshot.data_confidence_score,
                'status': 'success',
                'timestamp': datetime.now(UTC).isoformat(),
            }
            
            logger.info(f"Market Agent: Complete. TAM=${snapshot.tam_estimate.tam_usd:.0f}M")
            return result
        
        except Exception as e:
            logger.error(f"Market Agent failed: {str(e)}")
            return {
                'agent': 'market_agent',
                'drug': drug_name,
                'indication': indication,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now(UTC).isoformat(),
            }
    
    def _identify_gtm_risks(self, snapshot: MarketSnapshot) -> List[str]:
        """Identify go-to-market risks"""
        risks = []
        
        # Competitive risk
        critical_competitors = [c for c in snapshot.competitors if c.threat_level == CompetitorThreat.CRITICAL]
        if len(critical_competitors) > 2:
            risks.append("COMPETITIVE RISK: Crowded market with 3+ approved competitors limits pricing power")
        
        # Payer risk
        restricted_payers = [p for p in snapshot.payer_signals if p.reimbursement_status == ReimbursementStatus.RESTRICTED]
        if len(restricted_payers) > 1:
            risks.append(f"REIMBURSEMENT RISK: {len(restricted_payers)} major payers have coverage restrictions")
        
        # Market size risk
        if snapshot.tam_estimate and snapshot.tam_estimate.tam_usd < 500:
            risks.append("MARKET SIZE RISK: TAM <$500M limits blockbuster potential")
        
        # Timing risk
        if snapshot.revenue_scenarios:
            earliest_launch = min((s.launch_year for s in snapshot.revenue_scenarios), default=2025)
            if earliest_launch > 2026:
                risks.append("TIMING RISK: Delayed launch window provides competitors advantage")
        
        return risks


# ============================================================================
# Demo / Testing
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MARKET INTELLIGENCE AGENT - Demo")
    print("="*80 + "\n")
    
    agent = MarketAgent()
    
    # Run on sample drug + indication
    result = agent.run(
        drug_name="aspirin",
        indication="cardiovascular disease",
        options={"geography": "US"}
    )
    
    print(f"Status: {result['status']}")
    print(f"Drug: {result['drug']}")
    print(f"Indication: {result['indication']}")
    print(f"\nMarket Analysis:")
    print(f"TAM: ${result['tam_estimate']['tam_usd']:.0f}M")
    print(f"CAGR: {result['tam_estimate']['cagr_percent']}%")
    print(f"Market Phase: {result['market_phase']}")
    print(f"Data Confidence: {result['data_confidence']:.0%}\n")
    
    print("="*80)
    print("REVENUE SCENARIOS")
    print("="*80 + "\n")
    for scenario in result['revenue_scenarios']:
        print(f"{scenario['scenario_name']}:")
        print(f"  Launch: {scenario['launch_year']} | Peak Year: {scenario['peak_sales_year']}")
        print(f"  Peak Sales: ${scenario['peak_sales_usd']:.0f}M | Market Share: {scenario['market_share_at_peak']:.0%}")
        print(f"  5Y Revenue: ${scenario['five_year_revenue']:.0f}M | 10Y Revenue: ${scenario['ten_year_revenue']:.0f}M\n")
    
    print("="*80)
    print("COMPETITOR ANALYSIS")
    print("="*80 + "\n")
    for competitor in result['competitors']:
        print(f"{competitor['company_name']} - {competitor['drug_name']}")
        print(f"  Stage: {competitor['development_stage']}")
        print(f"  Market Share: {competitor['market_share_estimate']:.0%}")
        print(f"  Threat Level: {competitor['threat_level'].upper()}\n")
    
    print("="*80)
    print("KEY INSIGHTS & RISKS")
    print("="*80 + "\n")
    print("Insights:")
    for i, insight in enumerate(result['key_insights'], 1):
        print(f"  {i}. {insight}\n")
    
    print("Go-to-Market Risks:")
    for i, risk in enumerate(result['go_to_market_risks'], 1):
        print(f"  {i}. {risk}\n")
    
    print("="*80)
    print("MARKET SUMMARY")
    print("="*80 + "\n")
    print(result['market_snapshot']['market_summary'])
