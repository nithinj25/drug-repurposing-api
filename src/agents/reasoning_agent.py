"""
Reasoning Agent - Evidence Aggregation & Hypothesis Scoring

Aggregates multi-agent evidence, computes per-dimension scores and composite feasibility,
detects contradictions, and produces explainable, evidence-backed rankings.

Architecture:
1. Ingest & Normalize: Collect results from all worker agents
2. Evidence Store: Structured storage with provenance tracking
3. Feature Extraction: Transform evidence into scoring features
4. Constraint Checking: Apply safety/IP/regulatory vetoes
5. Scoring Engine: Hybrid rule-based + learned ensemble
6. Contradiction Detection: Identify conflicting evidence
7. Explainability: Generate human-readable justifications
8. Ranking: Sort and present top drug-indication hypotheses
"""

import os
import json
import logging
from datetime import datetime, timezone, UTC
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import math

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LangChain imports with fallback
try:
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class DecisionLevel(Enum):
    """Decision outcome levels"""
    REJECT = "reject"           # Hard veto (safety/IP)
    NOT_RECOMMENDED = "not_recommended"  # Low feasibility
    REVIEW_REQUIRED = "review_required"  # Moderate, needs expert review
    RECOMMENDED = "recommended"          # Good feasibility
    HIGHLY_RECOMMENDED = "highly_recommended"  # Excellent feasibility


class EvidenceType(Enum):
    """Types of evidence sources"""
    CLINICAL = "clinical"
    LITERATURE = "literature"
    PATENT = "patent"
    SAFETY = "safety"
    MOLECULAR = "molecular"
    MARKET = "market"
    REGULATORY = "regulatory"
    INTERNAL = "internal"


class DimensionType(Enum):
    """Scoring dimensions"""
    CLINICAL_EVIDENCE = "clinical_evidence"
    SAFETY_PROFILE = "safety_profile"
    PATENT_FREEDOM = "patent_freedom"
    MARKET_POTENTIAL = "market_potential"
    MOLECULAR_RATIONALE = "molecular_rationale"
    REGULATORY_PATH = "regulatory_path"


# Score weights for composite calculation
DIMENSION_WEIGHTS = {
    DimensionType.CLINICAL_EVIDENCE: 0.25,
    DimensionType.SAFETY_PROFILE: 0.25,
    DimensionType.PATENT_FREEDOM: 0.15,
    DimensionType.MARKET_POTENTIAL: 0.15,
    DimensionType.MOLECULAR_RATIONALE: 0.10,
    DimensionType.REGULATORY_PATH: 0.10,
}


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Evidence:
    """Individual piece of evidence from worker agents"""
    evidence_id: str
    source_agent: EvidenceType
    dimension: DimensionType
    content: str
    confidence: float  # 0-1
    polarity: str  # positive, negative, neutral
    metadata: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class DimensionScore:
    """Score for a specific dimension"""
    dimension: DimensionType
    score: float  # 0-1
    confidence: float  # 0-1
    evidence_count: int
    supporting_evidence: List[str]  # Evidence IDs
    contradicting_evidence: List[str]
    key_factors: List[str]
    explanation: str


@dataclass
class Contradiction:
    """Detected contradiction in evidence"""
    contradiction_id: str
    dimension: DimensionType
    evidence_a_id: str
    evidence_b_id: str
    description: str
    severity: str  # low, medium, high
    resolution_strategy: str


@dataclass
class Constraint:
    """Hard constraint/veto"""
    constraint_type: str  # safety, patent, regulatory
    is_violated: bool
    description: str
    blocking_evidence: List[str]


@dataclass
class Hypothesis:
    """Drug-indication repurposing hypothesis"""
    hypothesis_id: str
    drug_name: str
    indication: str
    composite_score: float  # 0-1
    decision: DecisionLevel
    dimension_scores: List[DimensionScore]
    constraints: List[Constraint]
    contradictions: List[Contradiction]
    all_evidence: List[Evidence]
    explanation: str
    confidence: float  # 0-1
    rank: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class ReasoningResult:
    """Final output from reasoning agent"""
    hypotheses: List[Hypothesis]
    total_evidence_count: int
    total_contradictions: int
    processing_time_ms: float
    metadata: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# ============================================================================
# EVIDENCE AGGREGATOR
# ============================================================================

class EvidenceAggregator:
    """Collect and normalize evidence from worker agents"""
    
    def __init__(self):
        self.evidence_store: Dict[str, List[Evidence]] = defaultdict(list)
        logger.info("EvidenceAggregator initialized")
    
    def ingest_clinical_evidence(self, drug: str, indication: str, clinical_result: Dict) -> List[Evidence]:
        """Process clinical agent results"""
        evidence_list = []
        
        # Extract trial evidence
        trials = clinical_result.get("trials", [])
        for trial in trials:
            # Efficacy evidence
            efficacy = trial.get("efficacy_summary", "")
            if efficacy:
                evidence_list.append(Evidence(
                    evidence_id=f"clin_eff_{trial.get('trial_id', 'unknown')}",
                    source_agent=EvidenceType.CLINICAL,
                    dimension=DimensionType.CLINICAL_EVIDENCE,
                    content=efficacy,
                    confidence=0.8 if trial.get("phase") in ["Phase 3", "Phase 4"] else 0.6,
                    polarity="positive" if "improved" in efficacy.lower() else "neutral",
                    metadata={"trial_id": trial.get("trial_id"), "phase": trial.get("phase")}
                ))
            
            # Safety evidence from trials
            safety_signals = trial.get("safety_signals", [])
            if safety_signals:
                formatted_signals = []
                for signal in safety_signals:
                    if isinstance(signal, dict):
                        term = signal.get("ae_term") or signal.get("term") or signal.get("name") or str(signal)
                        freq = signal.get("frequency") or signal.get("freq")
                        sev = signal.get("severity")
                        parts = [term]
                        if freq:
                            parts.append(f"freq={freq}")
                        if sev:
                            parts.append(f"sev={sev}")
                        formatted_signals.append(" | ".join(parts))
                    else:
                        formatted_signals.append(str(signal))

                evidence_list.append(Evidence(
                    evidence_id=f"clin_safe_{trial.get('trial_id', 'unknown')}",
                    source_agent=EvidenceType.CLINICAL,
                    dimension=DimensionType.SAFETY_PROFILE,
                    content=f"Safety signals: {', '.join(formatted_signals)}",
                    confidence=0.7,
                    polarity="negative" if safety_signals else "positive",
                    metadata={"trial_id": trial.get("trial_id"), "signals": safety_signals}
                ))
        
        return evidence_list
    
    def ingest_safety_evidence(self, drug: str, indication: str, safety_result: Dict) -> List[Evidence]:
        """Process safety agent results"""
        evidence_list = []
        
        # Overall safety score
        safety_score = safety_result.get("safety_score", 0.5)
        risk_level = safety_result.get("risk_level", "unknown")
        
        evidence_list.append(Evidence(
            evidence_id=f"safe_score_{drug}",
            source_agent=EvidenceType.SAFETY,
            dimension=DimensionType.SAFETY_PROFILE,
            content=f"Safety feasibility score: {safety_score:.2f} (Risk: {risk_level})",
            confidence=0.9,
            polarity="positive" if safety_score > 0.7 else ("negative" if safety_score < 0.4 else "neutral"),
            metadata={"score": safety_score, "risk_level": risk_level}
        ))
        
        # Red flags
        red_flags = safety_result.get("red_flags", [])
        for idx, flag in enumerate(red_flags):
            evidence_list.append(Evidence(
                evidence_id=f"safe_red_{drug}_{idx}",
                source_agent=EvidenceType.SAFETY,
                dimension=DimensionType.SAFETY_PROFILE,
                content=f"Critical safety concern: {flag}",
                confidence=0.95,
                polarity="negative",
                metadata={"flag_type": "red", "description": flag}
            ))
        
        # Amber flags
        amber_flags = safety_result.get("amber_flags", [])
        for idx, flag in enumerate(amber_flags):
            evidence_list.append(Evidence(
                evidence_id=f"safe_amber_{drug}_{idx}",
                source_agent=EvidenceType.SAFETY,
                dimension=DimensionType.SAFETY_PROFILE,
                content=f"Safety caution: {flag}",
                confidence=0.8,
                polarity="negative",
                metadata={"flag_type": "amber", "description": flag}
            ))
        
        return evidence_list
    
    def ingest_patent_evidence(self, drug: str, indication: str, patent_result: Dict) -> List[Evidence]:
        """Process patent agent results"""
        evidence_list = []
        
        # FTO analysis
        fto_score = patent_result.get("fto_score", 0.5)
        risk_assessment = patent_result.get("risk_assessment", "unknown")
        
        evidence_list.append(Evidence(
            evidence_id=f"pat_fto_{drug}_{indication}",
            source_agent=EvidenceType.PATENT,
            dimension=DimensionType.PATENT_FREEDOM,
            content=f"Freedom-to-operate score: {fto_score:.2f} ({risk_assessment})",
            confidence=0.85,
            polarity="positive" if fto_score > 0.7 else ("negative" if fto_score < 0.4 else "neutral"),
            metadata={"fto_score": fto_score, "risk": risk_assessment}
        ))
        
        # Blocking patents
        blocking_patents = patent_result.get("blocking_patents", [])
        for patent in blocking_patents:
            evidence_list.append(Evidence(
                evidence_id=f"pat_block_{patent.get('patent_id', 'unknown')}",
                source_agent=EvidenceType.PATENT,
                dimension=DimensionType.PATENT_FREEDOM,
                content=f"Blocking patent: {patent.get('title', 'Unknown')} (expires {patent.get('expiry', 'unknown')})",
                confidence=0.9,
                polarity="negative",
                metadata=patent
            ))
        
        return evidence_list
    
    def ingest_market_evidence(self, drug: str, indication: str, market_result: Dict) -> List[Evidence]:
        """Process market agent results"""
        evidence_list = []
        
        # Market size
        tam = market_result.get("tam_usd", 0)
        if tam:
            evidence_list.append(Evidence(
                evidence_id=f"mkt_tam_{drug}_{indication}",
                source_agent=EvidenceType.MARKET,
                dimension=DimensionType.MARKET_POTENTIAL,
                content=f"Total addressable market: ${tam:,.0f}",
                confidence=0.7,
                polarity="positive" if tam > 1e9 else "neutral",
                metadata={"tam_usd": tam}
            ))
        
        # Competitive intensity
        competitor_count = market_result.get("competitor_count", 0)
        evidence_list.append(Evidence(
            evidence_id=f"mkt_comp_{drug}_{indication}",
            source_agent=EvidenceType.MARKET,
            dimension=DimensionType.MARKET_POTENTIAL,
            content=f"Competitive landscape: {competitor_count} competitors",
            confidence=0.75,
            polarity="negative" if competitor_count > 10 else "neutral",
            metadata={"competitor_count": competitor_count}
        ))
        
        return evidence_list
    
    def ingest_literature_evidence(self, drug: str, indication: str, lit_result: Dict) -> List[Evidence]:
        """Process literature agent results"""
        evidence_list = []
        
        # Publication count
        pub_count = lit_result.get("publication_count", 0)
        evidence_list.append(Evidence(
            evidence_id=f"lit_count_{drug}_{indication}",
            source_agent=EvidenceType.LITERATURE,
            dimension=DimensionType.CLINICAL_EVIDENCE,
            content=f"Literature support: {pub_count} publications",
            confidence=0.6,
            polarity="positive" if pub_count > 10 else "neutral",
            metadata={"publication_count": pub_count}
        ))
        
        # Key findings
        findings = lit_result.get("key_findings", [])
        for idx, finding in enumerate(findings[:5]):
            evidence_list.append(Evidence(
                evidence_id=f"lit_find_{drug}_{indication}_{idx}",
                source_agent=EvidenceType.LITERATURE,
                dimension=DimensionType.MOLECULAR_RATIONALE,
                content=str(finding),  # ensure serialization-safe
                confidence=0.65,
                polarity="positive",
                metadata={"finding": finding}
            ))
        
        return evidence_list
    
    def aggregate_all(
        self,
        drug: str,
        indication: str,
        agent_results: Dict[str, Dict]
    ) -> List[Evidence]:
        """Aggregate evidence from all available agents"""
        all_evidence = []
        
        if "clinical" in agent_results:
            all_evidence.extend(self.ingest_clinical_evidence(drug, indication, agent_results["clinical"]))
        
        if "safety" in agent_results:
            all_evidence.extend(self.ingest_safety_evidence(drug, indication, agent_results["safety"]))
        
        if "patent" in agent_results:
            all_evidence.extend(self.ingest_patent_evidence(drug, indication, agent_results["patent"]))
        
        if "market" in agent_results:
            all_evidence.extend(self.ingest_market_evidence(drug, indication, agent_results["market"]))
        
        if "literature" in agent_results:
            all_evidence.extend(self.ingest_literature_evidence(drug, indication, agent_results["literature"]))
        
        logger.info(f"Aggregated {len(all_evidence)} evidence items for {drug} → {indication}")
        return all_evidence


# ============================================================================
# CONSTRAINT CHECKER
# ============================================================================

class ConstraintChecker:
    """Apply hard constraints and vetoes"""
    
    def __init__(self):
        self.veto_thresholds = {
            "safety_score": 0.3,  # Below this = veto
            "fto_score": 0.2,     # Below this = veto
        }
        logger.info("ConstraintChecker initialized")
    
    def check_safety_constraints(self, evidence: List[Evidence]) -> Constraint:
        """Check safety-related hard constraints"""
        safety_evidence = [e for e in evidence if e.dimension == DimensionType.SAFETY_PROFILE]
        
        # Extract safety score
        safety_score = 0.5
        for ev in safety_evidence:
            if "safety_score" in ev.evidence_id:
                safety_score = ev.metadata.get("score", 0.5)
                break
        
        # Check for red flags
        red_flags = [e for e in safety_evidence if "red" in e.evidence_id]
        
        is_violated = safety_score < self.veto_thresholds["safety_score"] or len(red_flags) >= 2
        
        return Constraint(
            constraint_type="safety",
            is_violated=is_violated,
            description=f"Safety score {safety_score:.2f} {'violates' if is_violated else 'meets'} minimum threshold {self.veto_thresholds['safety_score']}",
            blocking_evidence=[e.evidence_id for e in red_flags] if is_violated else []
        )
    
    def check_patent_constraints(self, evidence: List[Evidence]) -> Constraint:
        """Check patent-related hard constraints"""
        patent_evidence = [e for e in evidence if e.dimension == DimensionType.PATENT_FREEDOM]
        
        # Extract FTO score
        fto_score = 0.5
        for ev in patent_evidence:
            if "fto" in ev.evidence_id:
                fto_score = ev.metadata.get("fto_score", 0.5)
                break
        
        # Check for blocking patents
        blocking = [e for e in patent_evidence if "block" in e.evidence_id]
        
        is_violated = fto_score < self.veto_thresholds["fto_score"] or len(blocking) >= 3
        
        return Constraint(
            constraint_type="patent",
            is_violated=is_violated,
            description=f"FTO score {fto_score:.2f} {'violates' if is_violated else 'meets'} minimum threshold {self.veto_thresholds['fto_score']}",
            blocking_evidence=[e.evidence_id for e in blocking] if is_violated else []
        )
    
    def check_regulatory_constraints(self, evidence: List[Evidence]) -> Constraint:
        """Check regulatory constraints"""
        # Simplified - would check for regulatory blockers
        return Constraint(
            constraint_type="regulatory",
            is_violated=False,
            description="No regulatory blockers identified",
            blocking_evidence=[]
        )
    
    def check_all_constraints(self, evidence: List[Evidence]) -> List[Constraint]:
        """Check all hard constraints"""
        return [
            self.check_safety_constraints(evidence),
            self.check_patent_constraints(evidence),
            self.check_regulatory_constraints(evidence)
        ]


# ============================================================================
# SCORING ENGINE
# ============================================================================

class ScoringEngine:
    """Compute per-dimension and composite scores"""
    
    def __init__(self):
        logger.info("ScoringEngine initialized")
    
    def score_dimension(
        self,
        dimension: DimensionType,
        evidence: List[Evidence]
    ) -> DimensionScore:
        """Score a specific dimension"""
        dim_evidence = [e for e in evidence if e.dimension == dimension]
        
        if not dim_evidence:
            return DimensionScore(
                dimension=dimension,
                score=0.5,
                confidence=0.0,
                evidence_count=0,
                supporting_evidence=[],
                contradicting_evidence=[],
                key_factors=["No evidence available"],
                explanation=f"No evidence found for {dimension.value}"
            )
        
        # Compute weighted score
        positive_evidence = [e for e in dim_evidence if e.polarity == "positive"]
        negative_evidence = [e for e in dim_evidence if e.polarity == "negative"]
        
        # Weighted average based on confidence
        total_weight = sum(e.confidence for e in dim_evidence)
        if total_weight > 0:
            score = sum(
                e.confidence * (1.0 if e.polarity == "positive" else (0.0 if e.polarity == "negative" else 0.5))
                for e in dim_evidence
            ) / total_weight
        else:
            score = 0.5
        
        # Aggregate confidence
        avg_confidence = sum(e.confidence for e in dim_evidence) / len(dim_evidence)
        
        # Extract key factors
        key_factors = [e.content[:100] for e in sorted(dim_evidence, key=lambda x: x.confidence, reverse=True)[:3]]
        
        # Generate explanation
        explanation = self._generate_dimension_explanation(dimension, score, dim_evidence)
        
        return DimensionScore(
            dimension=dimension,
            score=score,
            confidence=avg_confidence,
            evidence_count=len(dim_evidence),
            supporting_evidence=[e.evidence_id for e in positive_evidence],
            contradicting_evidence=[e.evidence_id for e in negative_evidence],
            key_factors=key_factors,
            explanation=explanation
        )
    
    def _generate_dimension_explanation(
        self,
        dimension: DimensionType,
        score: float,
        evidence: List[Evidence]
    ) -> str:
        """Generate human-readable explanation for dimension score"""
        pos_count = sum(1 for e in evidence if e.polarity == "positive")
        neg_count = sum(1 for e in evidence if e.polarity == "negative")
        
        if score > 0.7:
            quality = "Strong"
        elif score > 0.5:
            quality = "Moderate"
        else:
            quality = "Weak"
        
        return (f"{quality} {dimension.value.replace('_', ' ')}: "
                f"Score {score:.2f} based on {len(evidence)} evidence items "
                f"({pos_count} supporting, {neg_count} contradicting)")
    
    def compute_composite_score(
        self,
        dimension_scores: List[DimensionScore],
        constraints: List[Constraint]
    ) -> Tuple[float, float]:
        """Compute composite feasibility score and confidence"""
        # Apply vetos first
        if any(c.is_violated for c in constraints):
            return 0.0, 1.0  # Zero score with high confidence if constraint violated
        
        # Weighted average of dimension scores
        total_weight = 0.0
        weighted_sum = 0.0
        confidence_sum = 0.0
        
        for dim_score in dimension_scores:
            weight = DIMENSION_WEIGHTS.get(dim_score.dimension, 0.1)
            weighted_sum += weight * dim_score.score
            total_weight += weight
            confidence_sum += dim_score.confidence
        
        composite = weighted_sum / total_weight if total_weight > 0 else 0.5
        avg_confidence = confidence_sum / len(dimension_scores) if dimension_scores else 0.0
        
        return composite, avg_confidence
    
    def score_all_dimensions(self, evidence: List[Evidence]) -> List[DimensionScore]:
        """Score all dimensions"""
        dimension_scores = []
        
        for dimension in DimensionType:
            dim_score = self.score_dimension(dimension, evidence)
            dimension_scores.append(dim_score)
        
        return dimension_scores


# ============================================================================
# CONTRADICTION DETECTOR
# ============================================================================

class ContradictionDetector:
    """Detect contradictions in evidence"""
    
    def __init__(self):
        self.llm = None
        self.use_llm = False
        
        # Try Groq first
        if LANGCHAIN_AVAILABLE and os.getenv("USE_GROQ") == "true" and os.getenv("GROQ_API_KEY"):
            try:
                self.llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.0,
                    api_key=os.getenv("GROQ_API_KEY")
                )
                self.use_llm = True
                logger.info("Using Groq for contradiction detection")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")
        
        if not self.use_llm:
            logger.warning("No LLM available, using rule-based contradiction detection")
    
    def detect_contradictions(self, evidence: List[Evidence]) -> List[Contradiction]:
        """Detect contradictions within evidence"""
        contradictions = []
        
        # Group evidence by dimension
        by_dimension = defaultdict(list)
        for ev in evidence:
            by_dimension[ev.dimension].append(ev)
        
        # Check each dimension for contradictions
        for dimension, dim_evidence in by_dimension.items():
            # Find opposing polarities
            positive = [e for e in dim_evidence if e.polarity == "positive"]
            negative = [e for e in dim_evidence if e.polarity == "negative"]
            
            # Flag if high-confidence opposing evidence exists
            for pos in positive:
                for neg in negative:
                    if pos.confidence > 0.7 and neg.confidence > 0.7:
                        contradictions.append(Contradiction(
                            contradiction_id=f"contr_{pos.evidence_id}_{neg.evidence_id}",
                            dimension=dimension,
                            evidence_a_id=pos.evidence_id,
                            evidence_b_id=neg.evidence_id,
                            description=f"Conflicting evidence: '{pos.content[:50]}...' vs '{neg.content[:50]}...'",
                            severity="high" if abs(pos.confidence - neg.confidence) < 0.1 else "medium",
                            resolution_strategy="Requires expert review to reconcile"
                        ))
        
        logger.info(f"Detected {len(contradictions)} contradictions")
        return contradictions


# ============================================================================
# EXPLAINABILITY MODULE
# ============================================================================

class ExplainabilityModule:
    """Generate human-readable explanations"""
    
    def __init__(self):
        self.llm = None
        self.use_llm = False
        
        # Try Groq first
        if LANGCHAIN_AVAILABLE and os.getenv("USE_GROQ") == "true" and os.getenv("GROQ_API_KEY"):
            try:
                self.llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    api_key=os.getenv("GROQ_API_KEY")
                )
                self.use_llm = True
                logger.info("Using Groq for explanation generation")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")
        
        if not self.use_llm:
            logger.warning("No LLM available, using template-based explanations")
    
    def explain_hypothesis(
        self,
        drug: str,
        indication: str,
        composite_score: float,
        decision: DecisionLevel,
        dimension_scores: List[DimensionScore],
        constraints: List[Constraint],
        contradictions: List[Contradiction]
    ) -> str:
        """Generate comprehensive explanation for hypothesis"""
        if self.use_llm:
            return self._llm_explain(drug, indication, composite_score, decision, 
                                    dimension_scores, constraints, contradictions)
        else:
            return self._template_explain(drug, indication, composite_score, decision,
                                         dimension_scores, constraints, contradictions)
    
    def _llm_explain(
        self,
        drug: str,
        indication: str,
        composite_score: float,
        decision: DecisionLevel,
        dimension_scores: List[DimensionScore],
        constraints: List[Constraint],
        contradictions: List[Contradiction]
    ) -> str:
        """LLM-generated explanation"""
        # Prepare dimension summary
        dim_summary = "\n".join([
            f"- {ds.dimension.value.replace('_', ' ').title()}: {ds.score:.2f} ({ds.explanation})"
            for ds in sorted(dimension_scores, key=lambda x: x.score, reverse=True)[:4]
        ])
        
        # Constraint summary
        violated = [c for c in constraints if c.is_violated]
        constraint_summary = "\n".join([f"- VIOLATED: {c.description}" for c in violated]) if violated else "All constraints satisfied"
        
        # Contradiction summary
        contr_summary = f"{len(contradictions)} contradictions detected" if contradictions else "No contradictions"
        
        prompt = f"""Generate a concise 3-4 sentence explanation for this drug repurposing hypothesis.

Drug: {drug}
Indication: {indication}
Composite Score: {composite_score:.2f}/1.0
Decision: {decision.value.replace('_', ' ').upper()}

Top Dimension Scores:
{dim_summary}

Constraints:
{constraint_summary}

Contradictions: {contr_summary}

Explain: Why this score? What are the key strengths and concerns? What's the overall recommendation?
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM explanation failed: {e}")
            return self._template_explain(drug, indication, composite_score, decision,
                                         dimension_scores, constraints, contradictions)
    
    def _template_explain(
        self,
        drug: str,
        indication: str,
        composite_score: float,
        decision: DecisionLevel,
        dimension_scores: List[DimensionScore],
        constraints: List[Constraint],
        contradictions: List[Contradiction]
    ) -> str:
        """Template-based explanation"""
        explanation = f"Repurposing {drug} for {indication}: "
        
        # Overall assessment
        if decision == DecisionLevel.HIGHLY_RECOMMENDED:
            explanation += f"HIGHLY RECOMMENDED (score: {composite_score:.2f}). "
        elif decision == DecisionLevel.RECOMMENDED:
            explanation += f"RECOMMENDED (score: {composite_score:.2f}). "
        elif decision == DecisionLevel.REVIEW_REQUIRED:
            explanation += f"REQUIRES EXPERT REVIEW (score: {composite_score:.2f}). "
        elif decision == DecisionLevel.NOT_RECOMMENDED:
            explanation += f"NOT RECOMMENDED (score: {composite_score:.2f}). "
        else:
            explanation += f"REJECTED (score: {composite_score:.2f}). "
        
        # Top strengths
        top_dims = sorted(dimension_scores, key=lambda x: x.score, reverse=True)[:2]
        if top_dims[0].score > 0.6:
            explanation += f"Strengths: {top_dims[0].dimension.value.replace('_', ' ')} ({top_dims[0].score:.2f})"
            if len(top_dims) > 1 and top_dims[1].score > 0.6:
                explanation += f", {top_dims[1].dimension.value.replace('_', ' ')} ({top_dims[1].score:.2f})"
            explanation += ". "
        
        # Key concerns
        violated = [c for c in constraints if c.is_violated]
        if violated:
            explanation += f"CRITICAL: {violated[0].description}. "
        
        weak_dims = [ds for ds in dimension_scores if ds.score < 0.4]
        if weak_dims and not violated:
            explanation += f"Concerns: {weak_dims[0].dimension.value.replace('_', ' ')} score {weak_dims[0].score:.2f}. "
        
        # Contradictions
        if contradictions:
            explanation += f"Note: {len(contradictions)} evidence contradictions require resolution."
        
        return explanation


# ============================================================================
# REASONING AGENT (MAIN ORCHESTRATOR)
# ============================================================================

class ReasoningAgent:
    """
    Main reasoning agent that orchestrates:
    1. Evidence aggregation from worker agents
    2. Feature extraction and normalization
    3. Constraint checking (safety/IP vetoes)
    4. Multi-dimensional scoring
    5. Contradiction detection
    6. Explainability generation
    7. Hypothesis ranking
    """
    
    def __init__(self):
        self.aggregator = EvidenceAggregator()
        self.constraint_checker = ConstraintChecker()
        self.scoring_engine = ScoringEngine()
        self.contradiction_detector = ContradictionDetector()
        self.explainability = ExplainabilityModule()
        
        logger.info("ReasoningAgent initialized with all components")
    
    def process_hypothesis(
        self,
        drug: str,
        indication: str,
        agent_results: Dict[str, Dict]
    ) -> Hypothesis:
        """Process a single drug-indication hypothesis"""
        logger.info(f"Processing hypothesis: {drug} → {indication}")
        
        # 1. Aggregate evidence
        evidence = self.aggregator.aggregate_all(drug, indication, agent_results)
        
        # 2. Check constraints
        constraints = self.constraint_checker.check_all_constraints(evidence)
        
        # 3. Score dimensions
        dimension_scores = self.scoring_engine.score_all_dimensions(evidence)
        
        # 4. Compute composite score
        composite_score, confidence = self.scoring_engine.compute_composite_score(
            dimension_scores, constraints
        )
        
        # 5. Detect contradictions
        contradictions = self.contradiction_detector.detect_contradictions(evidence)
        
        # 6. Determine decision level
        decision = self._determine_decision(composite_score, constraints)
        
        # 7. Generate explanation
        explanation = self.explainability.explain_hypothesis(
            drug, indication, composite_score, decision,
            dimension_scores, constraints, contradictions
        )
        
        # 8. Construct hypothesis
        hypothesis = Hypothesis(
            hypothesis_id=f"hyp_{drug}_{indication}_{datetime.now(UTC).timestamp()}",
            drug_name=drug,
            indication=indication,
            composite_score=composite_score,
            decision=decision,
            dimension_scores=dimension_scores,
            constraints=constraints,
            contradictions=contradictions,
            all_evidence=evidence,
            explanation=explanation,
            confidence=confidence
        )
        
        logger.info(f"Hypothesis processed: {decision.value} (score: {composite_score:.2f})")
        return hypothesis
    
    def _determine_decision(
        self,
        composite_score: float,
        constraints: List[Constraint]
    ) -> DecisionLevel:
        """Determine decision level based on score and constraints"""
        # Hard vetoes
        if any(c.is_violated for c in constraints):
            return DecisionLevel.REJECT
        
        # Score-based decisions
        if composite_score >= 0.8:
            return DecisionLevel.HIGHLY_RECOMMENDED
        elif composite_score >= 0.65:
            return DecisionLevel.RECOMMENDED
        elif composite_score >= 0.45:
            return DecisionLevel.REVIEW_REQUIRED
        elif composite_score >= 0.25:
            return DecisionLevel.NOT_RECOMMENDED
        else:
            return DecisionLevel.REJECT
    
    def rank_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Rank hypotheses by composite score"""
        ranked = sorted(hypotheses, key=lambda h: h.composite_score, reverse=True)
        
        for idx, hyp in enumerate(ranked, 1):
            hyp.rank = idx
        
        return ranked
    
    def run(
        self,
        hypotheses_data: List[Dict[str, Any]]
    ) -> ReasoningResult:
        """
        Main entry point for reasoning agent
        
        Args:
            hypotheses_data: List of dicts with keys:
                - drug: str
                - indication: str
                - agent_results: Dict[agent_name, result_dict]
        
        Returns:
            ReasoningResult with ranked hypotheses
        """
        import time
        start_time = time.time()
        
        logger.info(f"Processing {len(hypotheses_data)} hypotheses")
        
        hypotheses = []
        for hyp_data in hypotheses_data:
            try:
                hypothesis = self.process_hypothesis(
                    drug=hyp_data["drug"],
                    indication=hyp_data["indication"],
                    agent_results=hyp_data.get("agent_results", {})
                )
                hypotheses.append(hypothesis)
            except Exception as e:
                logger.error(f"Failed to process hypothesis {hyp_data.get('drug')} → {hyp_data.get('indication')}: {e}")
                continue
        
        # Rank hypotheses
        ranked_hypotheses = self.rank_hypotheses(hypotheses)
        
        # Compute statistics
        total_evidence = sum(len(h.all_evidence) for h in hypotheses)
        total_contradictions = sum(len(h.contradictions) for h in hypotheses)
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        result = ReasoningResult(
            hypotheses=ranked_hypotheses,
            total_evidence_count=total_evidence,
            total_contradictions=total_contradictions,
            processing_time_ms=processing_time,
            metadata={
                "hypothesis_count": len(hypotheses),
                "highly_recommended": sum(1 for h in hypotheses if h.decision == DecisionLevel.HIGHLY_RECOMMENDED),
                "recommended": sum(1 for h in hypotheses if h.decision == DecisionLevel.RECOMMENDED),
                "review_required": sum(1 for h in hypotheses if h.decision == DecisionLevel.REVIEW_REQUIRED),
                "rejected": sum(1 for h in hypotheses if h.decision == DecisionLevel.REJECT)
            }
        )
        
        logger.info(f"Reasoning complete: {len(ranked_hypotheses)} hypotheses ranked in {processing_time:.1f}ms")
        return result
    
    def export_results(self, result: ReasoningResult, output_path: str):
        """Export results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        logger.info(f"Results exported to {output_path}")


# ============================================================================
# MAIN EXECUTION & DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("REASONING AGENT - Evidence Aggregation & Hypothesis Scoring")
    print("="*80 + "\n")
    
    # Initialize agent
    agent = ReasoningAgent()
    
    # Mock data for demonstration
    mock_hypotheses = [
        {
            "drug": "aspirin",
            "indication": "cardiovascular disease",
            "agent_results": {
                "safety": {
                    "safety_score": 0.90,
                    "risk_level": "green",
                    "red_flags": [],
                    "amber_flags": []
                },
                "clinical": {
                    "trials": [
                        {
                            "trial_id": "NCT00000001",
                            "phase": "Phase 3",
                            "efficacy_summary": "Demonstrated 25% reduction in cardiovascular events",
                            "safety_signals": []
                        }
                    ]
                },
                "patent": {
                    "fto_score": 0.95,
                    "risk_assessment": "low",
                    "blocking_patents": []
                },
                "market": {
                    "tam_usd": 15e9,
                    "competitor_count": 5
                },
                "literature": {
                    "publication_count": 250,
                    "key_findings": [
                        "Strong anti-platelet effects reduce thrombosis",
                        "Well-established safety profile over decades"
                    ]
                }
            }
        },
        {
            "drug": "metformin",
            "indication": "Alzheimer's disease",
            "agent_results": {
                "safety": {
                    "safety_score": 0.85,
                    "risk_level": "green",
                    "red_flags": [],
                    "amber_flags": ["GI side effects common"]
                },
                "clinical": {
                    "trials": [
                        {
                            "trial_id": "NCT00000002",
                            "phase": "Phase 2",
                            "efficacy_summary": "Preliminary cognitive improvement noted",
                            "safety_signals": ["nausea"]
                        }
                    ]
                },
                "patent": {
                    "fto_score": 0.85,
                    "risk_assessment": "low",
                    "blocking_patents": []
                },
                "market": {
                    "tam_usd": 50e9,
                    "competitor_count": 12
                },
                "literature": {
                    "publication_count": 45,
                    "key_findings": [
                        "AMPK activation may reduce amyloid burden",
                        "Epidemiological data suggests protective effect"
                    ]
                }
            }
        }
    ]
    
    # Process hypotheses
    result = agent.run(mock_hypotheses)
    
    # Display results
    print(f"Processed {len(result.hypotheses)} hypotheses in {result.processing_time_ms:.1f}ms\n")
    print(f"Total Evidence Items: {result.total_evidence_count}")
    print(f"Total Contradictions: {result.total_contradictions}\n")
    
    print("="*80)
    print("RANKED HYPOTHESES")
    print("="*80 + "\n")
    
    for hyp in result.hypotheses:
        print(f"#{hyp.rank} - {hyp.drug_name} → {hyp.indication}")
        print(f"   Score: {hyp.composite_score:.3f} | Decision: {hyp.decision.value.upper()}")
        print(f"   Confidence: {hyp.confidence:.2f} | Evidence: {len(hyp.all_evidence)} items")
        print(f"\n   {hyp.explanation}\n")
        
        # Show dimension breakdown
        print("   Dimension Scores:")
        for ds in sorted(hyp.dimension_scores, key=lambda x: x.score, reverse=True)[:4]:
            print(f"      • {ds.dimension.value.replace('_', ' ').title()}: {ds.score:.2f}")
        
        # Show constraints
        violated = [c for c in hyp.constraints if c.is_violated]
        if violated:
            print(f"\n   ⚠️  CONSTRAINTS VIOLATED:")
            for c in violated:
                print(f"      • {c.description}")
        
        # Show contradictions
        if hyp.contradictions:
            print(f"\n   ⚠️  CONTRADICTIONS ({len(hyp.contradictions)}):")
            for contr in hyp.contradictions[:2]:
                print(f"      • {contr.description}")
        
        print("\n" + "-"*80 + "\n")
    
    # Export results
    output_file = "reasoning_results.json"
    agent.export_results(result, output_file)
    print(f"✅ Full results exported to: {output_file}\n")
    print("="*80 + "\n")
