from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, UTC
import uuid
import logging
import sys
from pathlib import Path
import importlib

# Ensure project root is on path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Local orchestrated reasoning
from src.agents.reasoning_agent import ReasoningAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models & Enums
# ============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class EvidenceDimension(str, Enum):
    CLINICAL = "clinical"
    LITERATURE = "literature"
    SAFETY = "safety"
    PATENT = "patent"
    MOLECULAR = "molecular"
    MARKET = "market"
    INTERNAL = "internal"


@dataclass
class DrugIndicationQuery:
    """User input: drug + indication + options"""
    drug_name: str
    indication: str
    drug_synonyms: List[str] = field(default_factory=list)
    indication_synonyms: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Task:
    """Atomic task dispatched to an agent"""
    task_id: str
    agent_name: str
    dimension: EvidenceDimension
    query: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        data['status'] = self.status.value
        data['dimension'] = self.dimension.value
        return data


@dataclass
class JobMetadata:
    """Overall job tracking"""
    job_id: str
    created_at: datetime
    user_id: str
    query: DrugIndicationQuery
    tasks: Dict[str, Task] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    human_review_required: bool = False
    human_review_paused_at: Optional[datetime] = None
    reasoning_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        data = {
            'job_id': self.job_id,
            'created_at': self.created_at.isoformat(),
            'user_id': self.user_id,
            'query': self.query.to_dict(),
            'status': self.status.value,
            'human_review_required': self.human_review_required,
            'human_review_paused_at': self.human_review_paused_at.isoformat() if self.human_review_paused_at else None,
            'reasoning_result': self.reasoning_result,
            'tasks': {k: v.to_dict() for k, v in self.tasks.items()},
        }
        return data


# ============================================================================
# Query Normalizer
# ============================================================================

class QueryNormalizer:
    """Resolves drug synonyms and canonicalizes indication text"""
    
    def __init__(self):
        # In production, load from DrugBank/ChEMBL
        self.drug_synonyms_map = {
            "aspirin": ["acetylsalicylic acid", "asa", "bayer"],
            "metformin": ["glucophage", "fortamet"],
            "ibuprofen": ["advil", "motrin", "nurofen"],
        }
        self.indication_map = {
            "diabetes": ["type 2 diabetes", "diabetes mellitus", "t2dm"],
            "hypertension": ["high blood pressure", "htn", "hypertensive disease"],
            "inflammation": ["inflammatory condition", "inflamm"],
        }
    
    def normalize_drug(self, drug_name: str) -> str:
        """Return canonical drug name"""
        return drug_name.lower().strip()
    
    def normalize_indication(self, indication: str) -> str:
        """Return canonical indication text"""
        return indication.lower().strip()
    
    def expand_synonyms(self, drug_name: str) -> List[str]:
        """Return list of known synonyms"""
        canonical = self.normalize_drug(drug_name)
        return self.drug_synonyms_map.get(canonical, [canonical])


# ============================================================================
# Task Planner
# ============================================================================

class TaskPlanner:
    """Decomposes query into deterministic task templates"""
    
    AGENT_TO_DIMENSION = {
        "literature_agent": EvidenceDimension.LITERATURE,
        "clinical_agent": EvidenceDimension.CLINICAL,
        "safety_agent": EvidenceDimension.SAFETY,
        "patent_agent": EvidenceDimension.PATENT,
        "molecular_agent": EvidenceDimension.MOLECULAR,
        "market_agent": EvidenceDimension.MARKET,
        "internal_agent": EvidenceDimension.INTERNAL,
    }
    
    def plan_tasks(self, query: DrugIndicationQuery) -> List[Task]:
        """Create task list based on query options"""
        tasks = []
        
        # Always run core agents
        core_agents = [
            "literature_agent",
            "clinical_agent",
            "safety_agent",
            "molecular_agent",
        ]
        
        # Optionally add patent, market, internal
        if query.options.get("include_patent", True):
            core_agents.append("patent_agent")
        if query.options.get("include_market", True):
            core_agents.append("market_agent")
        if query.options.get("use_internal_data", False):
            core_agents.append("internal_agent")
        
        for agent_name in core_agents:
            dimension = self.AGENT_TO_DIMENSION[agent_name]
            task = Task(
                task_id=str(uuid.uuid4()),
                agent_name=agent_name,
                dimension=dimension,
                query=f"Analyze {query.drug_name} for {query.indication}"
            )
            tasks.append(task)
        
        return tasks


# ============================================================================
# Result Aggregator
# ============================================================================

class ResultAggregator:
    """Collects and validates agent outputs"""
    
    def aggregate(self, tasks: Dict[str, Task]) -> Dict[str, Any]:
        """Merge all task results into structured evidence"""
        aggregated = {
            'by_dimension': {},
            'raw_evidence': [],
            'validation_issues': [],
        }
        
        for task_id, task in tasks.items():
            if task.status == TaskStatus.COMPLETED and task.result:
                dimension_name = task.dimension.value
                if dimension_name not in aggregated['by_dimension']:
                    aggregated['by_dimension'][dimension_name] = []
                
                aggregated['by_dimension'][dimension_name].append({
                    'task_id': task.task_id,
                    'agent': task.agent_name,
                    'result': task.result,
                })
                aggregated['raw_evidence'].append(task.result)
            
            elif task.status == TaskStatus.FAILED:
                aggregated['validation_issues'].append({
                    'task_id': task.task_id,
                    'agent': task.agent_name,
                    'error': task.error,
                })
        
        return aggregated


# ============================================================================
# Master Agent (Coordinator)
# ============================================================================

class MasterAgent:
    """
    Orchestrator that:
    1. Accepts user query (drug + indication + options)
    2. Normalizes the query
    3. Plans tasks for specialized agents
    4. Dispatches tasks (via message broker)
    5. Collects results
    6. Validates outputs
    7. Coordinates human-in-loop checkpoints
    8. Triggers final synthesis (Reasoning Agent → ReportGenerator)
    """
    
    def __init__(self, user_id: str = "demo_user"):
        self.user_id = user_id
        self.normalizer = QueryNormalizer()
        self.planner = TaskPlanner()
        self.aggregator = ResultAggregator()
        self.job_store: Dict[str, JobMetadata] = {}
        self.task_results: Dict[str, Any] = {}
        self.reasoning_agent = ReasoningAgent()
        # Agent registry: module path -> class name
        self.agent_registry: Dict[str, Dict[str, str]] = {
            "clinical_agent": {"module": "src.agents.clinical_agent", "class": "ClinicalTrialsAgent"},
            "literature_agent": {"module": "src.agents.literature_agent", "class": "LiteratureAgent"},
            "safety_agent": {"module": "src.agents.safety_agent", "class": "SafetyAgent"},
            "patent_agent": {"module": "src.agents.patent_agent", "class": "PatentAgent"},
            "market_agent": {"module": "src.agents.market_agent", "class": "MarketAgent"},
            "molecular_agent": {"module": "src.agents.molecular_agent", "class": "MolecularAgent"},
        }
        logger.info("MasterAgent initialized")
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    def start_job(self, drug_name: str, indication: str, options: Optional[Dict] = None) -> str:
        """
        Entry point: user submits a query.
        Returns: job_id for polling/status checks
        """
        if options is None:
            options = {}
        
        # 1. Normalize query
        normalized_drug = self.normalizer.normalize_drug(drug_name)
        normalized_indication = self.normalizer.normalize_indication(indication)
        drug_synonyms = self.normalizer.expand_synonyms(drug_name)
        
        query = DrugIndicationQuery(
            drug_name=normalized_drug,
            indication=normalized_indication,
            drug_synonyms=drug_synonyms,
            options=options
        )
        
        # 2. Create job metadata
        job_id = str(uuid.uuid4())
        job = JobMetadata(
            job_id=job_id,
            created_at=datetime.now(UTC),
            user_id=self.user_id,
            query=query,
        )
        
        # 3. Plan tasks
        tasks = self.planner.plan_tasks(query)
        for task in tasks:
            job.tasks[task.task_id] = task
        
        # 4. Store job
        self.job_store[job_id] = job
        logger.info(f"Job {job_id} created with {len(tasks)} tasks")
        
        # 5. Dispatch tasks
        self._dispatch_tasks(job_id)
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Retrieve current job status - returns COMPLETE information including all agent results"""
        if job_id not in self.job_store:
            return {"error": "Job not found"}
        
        job = self.job_store[job_id]
        
        # Build tasks dictionary with all results
        tasks_dict = {}
        for task_id, task in job.tasks.items():
            tasks_dict[task_id] = {
                'task_id': task.task_id,
                'agent_name': task.agent_name,
                'dimension': task.dimension.value,
                'status': task.status.value,
                'result': task.result,
                'error': task.error,
                'created_at': task.created_at.isoformat(),
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            }
        
        return {
            'job_id': job_id,
            'drug_name': job.query.drug_name,
            'indication': job.query.indication,
            'status': job.status.value,
            'created_at': job.created_at.isoformat(),
            'query': job.query.to_dict(),
            'task_summary': {
                'total': len(job.tasks),
                'completed': sum(1 for t in job.tasks.values() if t.status == TaskStatus.COMPLETED),
                'failed': sum(1 for t in job.tasks.values() if t.status == TaskStatus.FAILED),
                'pending': sum(1 for t in job.tasks.values() if t.status == TaskStatus.PENDING),
            },
            'tasks': tasks_dict,  # Include ALL task results
            'reasoning_result': job.reasoning_result,  # Include reasoning/synthesis result
            'human_review_required': job.human_review_required,
        }
    
    def submit_task_result(self, job_id: str, task_id: str, result: Dict[str, Any], success: bool = True):
        """Agent callback: submit task result"""
        if job_id not in self.job_store:
            logger.error(f"Job {job_id} not found")
            return
        
        job = self.job_store[job_id]
        if task_id not in job.tasks:
            logger.error(f"Task {task_id} not found in job {job_id}")
            return
        
        task = job.tasks[task_id]
        
        if success:
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now(UTC)
            logger.info(f"Task {task_id} completed successfully")
        else:
            task.status = TaskStatus.FAILED
            task.error = result.get('error', 'Unknown error')
            task.completed_at = datetime.now(UTC)
            logger.warning(f"Task {task_id} failed: {task.error}")
            
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.info(f"Task {task_id} queued for retry ({task.retry_count}/{task.max_retries})")
        
        self._check_job_completion(job_id)
    
    def approve_human_review(self, job_id: str) -> Dict[str, Any]:
        """User approves job to proceed to synthesis"""
        if job_id not in self.job_store:
            return {"error": "Job not found"}
        
        job = self.job_store[job_id]
        job.human_review_required = False
        logger.info(f"Job {job_id} approved by human reviewer")
        
        return self._trigger_synthesis(job_id)
    
    # ========================================================================
    # Internal Methods
    # ========================================================================
    
    def _dispatch_tasks(self, job_id: str):
        """Send tasks to agents"""
        job = self.job_store[job_id]
        
        for task in job.tasks.values():
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now(UTC)
            logger.info(f"Dispatching task {task.task_id} to {task.agent_name}")
            try:
                result = self._execute_task(task, job.query)
                self.submit_task_result(job_id, task.task_id, result, success=True)
            except Exception as e:
                logger.exception(f"Task {task.task_id} failed: {e}")
                self.submit_task_result(job_id, task.task_id, {"error": str(e)}, success=False)
    
    def _check_job_completion(self, job_id: str):
        """Check if all tasks done; trigger synthesis if so"""
        job = self.job_store[job_id]
        
        total = len(job.tasks)
        completed = sum(1 for t in job.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in job.tasks.values() if t.status == TaskStatus.FAILED)
        
        if completed + failed == total:
            job.status = TaskStatus.COMPLETED
            logger.info(f"Job {job_id}: All tasks completed ({completed} success, {failed} failed)")
            
            if failed > 0:
                job.human_review_required = True
                job.human_review_paused_at = datetime.now(UTC)
                logger.warning(f"Job {job_id} requires human review due to {failed} failed tasks")
            else:
                self._trigger_synthesis(job_id)
    
    def _trigger_synthesis(self, job_id: str) -> Dict[str, Any]:
        """Aggregate evidence and call Reasoning Agent"""
        job = self.job_store[job_id]
        
        aggregated = self.aggregator.aggregate(job.tasks)
        logger.info(f"Job {job_id}: Evidence aggregated, {len(aggregated['raw_evidence'])} items")
        
        # Build reasoning input (single hypothesis for this job)
        reasoning_input = [self._build_reasoning_payload(job, aggregated)]
        reasoning_result = self.reasoning_agent.run(reasoning_input)
        job.reasoning_result = asdict(reasoning_result)
        
        return {
            'job_id': job_id,
            'status': 'synthesis_complete',
            'evidence_summary': {
                'total_items': len(aggregated['raw_evidence']),
                'by_dimension': {k: len(v) for k, v in aggregated['by_dimension'].items()},
                'validation_issues': len(aggregated['validation_issues']),
            },
            'reasoning_result': job.reasoning_result,
        }

    def _execute_task(self, task: Task, query: DrugIndicationQuery) -> Dict[str, Any]:
        """Run the appropriate agent synchronously and return its result"""
        if task.agent_name not in self.agent_registry:
            raise ValueError(f"Unknown agent: {task.agent_name}")
        meta = self.agent_registry[task.agent_name]
        module = importlib.import_module(meta["module"])
        agent_cls = getattr(module, meta["class"])
        agent = agent_cls()
        if not hasattr(agent, "run"):
            raise AttributeError(f"Agent {task.agent_name} missing run() method")
        result = agent.run(drug_name=query.drug_name, indication=query.indication)
        return self._serialize_result(result)

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """Best-effort conversion of agent output to dict"""
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        if hasattr(result, "dict"):
            return result.dict()  # pydantic-style
        if hasattr(result, "to_dict"):
            return result.to_dict()
        if hasattr(result, "__dataclass_fields__"):
            return asdict(result)
        if hasattr(result, "__dict__"):
            return dict(result.__dict__)
        raise TypeError(f"Unsupported result type: {type(result)}")

    def _build_reasoning_payload(self, job: JobMetadata, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Convert aggregated task results into ReasoningAgent input structure"""
        agent_results: Dict[str, Any] = {}
        
        # Pick the first completed result per dimension (simple heuristic)
        for dimension, results in aggregated['by_dimension'].items():
            if not results:
                continue
            # If multiple, prefer the one with highest evidence_count if present
            selected = max(results, key=lambda r: r['result'].get('evidence_count', 0))
            agent_results[dimension] = selected['result']
        
        return {
            'drug': job.query.drug_name,
            'indication': job.query.indication,
            'agent_results': agent_results,
        }


if __name__ == "__main__":
    master = MasterAgent(user_id="demo_user")
    
    # Start a job
    job_id = master.start_job(
        drug_name="ibuprofen",
        indication="inflammatory bowel disease",
        options={"include_patent": True, "use_internal_data": False}
    )
    print(f"\n✓ Job started: {job_id}")
    
    # All agents run synchronously during start_job; show final status
    final_status = master.get_job_status(job_id)
    print(f"\n✓ Final job status:\n{final_status}")
    
    if master.job_store[job_id].reasoning_result:
        print("\n✓ Reasoning result (composite ranking):")
        print(master.job_store[job_id].reasoning_result)
