from datetime import datetime, UTC
from typing import Dict, List, Optional


class MolecularAgent:
    """Lightweight mechanistic heuristic so orchestration can complete without GPU/LLM."""

    def __init__(self):
        # Curated minimal facts to avoid remote calls while keeping outputs meaningful.
        self._knowledge_base: Dict[str, Dict[str, List[str]]] = {
            "aspirin": {
                "targets": ["PTGS1 (COX-1)", "PTGS2 (COX-2)", "TBXAS1"],
                "pathways": [
                    "Arachidonic acid metabolism",
                    "Platelet activation",
                    "Prostaglandin synthesis"
                ],
            },
            "metformin": {
                "targets": ["AMPK", "mGPD"],
                "pathways": ["Gluconeogenesis", "AMPK signaling"],
            },
        }

    def analyze_structure(self, drug_name: str) -> Dict[str, List[str]]:
        key = drug_name.lower()
        if key in self._knowledge_base:
            return self._knowledge_base[key]

        # Fallback heuristic for unknown compounds.
        return {
            "targets": ["Unknown target"],
            "pathways": ["Mechanism requires wet lab validation"],
        }

    def list_targets(self, drug_name: str) -> List[str]:
        return self.analyze_structure(drug_name).get("targets", [])

    def summarize_mechanism(self, drug_name: str, indication: str, targets: List[str], pathways: List[str]) -> str:
        target_str = ", ".join(targets) if targets else "unspecified targets"
        pathway_str = ", ".join(pathways) if pathways else "unspecified pathways"
        return (
            f"{drug_name.title()} may influence {indication.lower()} via {target_str} across {pathway_str}. "
            "This is a rule-based summary; validate with experiments and literature."
        )

    def run(self, drug_name: str, indication: str, chemical_structure: Optional[str] = None, bioactivity_data: Optional[dict] = None, pathway_data: Optional[dict] = None) -> dict:
        # Use lightweight heuristics to keep the pipeline fast and deterministic.
        targets = self.list_targets(drug_name)
        pathways = self.analyze_structure(drug_name).get("pathways", [])
        mechanism_summary = self.summarize_mechanism(drug_name, indication, targets, pathways)

        return {
            "drug": drug_name,
            "indication": indication,
            "predicted_targets": targets,
            "pathways": pathways,
            "mechanistic_plausibility": "moderate" if targets else "unknown",
            "safety_flags": [],
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": mechanism_summary,
            "method": "rule_based_stub",
        }