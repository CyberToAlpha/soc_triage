"""
Core models and configuration for the SOC Triage Environment.

This module defines the required data structures (actions, observations, states),
the configuration, and the trajectory evaluation logic. It is strictly typed
using Pydantic to ensure seamless interoperability with the OpenEnv framework.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """
    Application and setup configuration for the environment.
    """
    enable_web_interface: bool = True
    max_episode_steps: int = 100


class SocTriageObservation(BaseModel):
    """
    Represents a simulated JSON network log event passed to the agent.
    """
    timestamp: float = Field(..., description="Unix timestamp of the event.")
    source_ip: str = Field(..., description="The source IP address of the traffic.")
    request_payload: str = Field(..., description="The payload or URI of the request.")
    http_status: int = Field(default=200, description="HTTP response status code.")
    user_agent: str = Field(default="unknown", description="User-Agent header value.")
    
    # OpenEnv standard step properties
    reward: float = Field(default=0.0, description="Reward accumulated so far.")
    done: bool = Field(default=False, description="Flag indicating if the episode is finished.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context.")


ActionType = Literal["block_ip", "allow_traffic", "flag_for_review"]


class SocTriageAction(BaseModel):
    """
    An action taken by the SOC analyst against a specific network target.
    
    Exactly 3 target actions are supported.
    """
    action_type: ActionType = Field(..., description="The type of triage action to apply.")
    target_ip: Optional[str] = Field(default=None, description="The IP address to target.")
    target_log_index: Optional[int] = Field(default=None, description="Optional index of the specific log.")
    rationale: Optional[str] = Field(default=None, description="Analyst reasoning for the action.")


class TaskDifficulty(str, Enum):
    """Difficulty levels for predefined triage scenarios."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskDefinition(BaseModel):
    """
    Machine-readable definition of a SOC scenario's parameters and ground truth.
    """
    difficulty: TaskDifficulty
    description: str
    attack_pattern: str
    noise_level: Literal["low", "medium", "high"]
    trajectory_complexity: Literal["single_event", "short_sequence", "long_horizon_correlations"]
    
    # Ground truth for grading
    malicious_ips: Set[str] = Field(default_factory=set)
    benign_ips: Set[str] = Field(default_factory=set)
    suspicious_ips: Set[str] = Field(default_factory=set)


# -------------------------------------------------------------------------
# The 3 supported triage tasks
# -------------------------------------------------------------------------

EASY_TASK = TaskDefinition(
    difficulty=TaskDifficulty.EASY,
    description="Brute Force attack",
    attack_pattern="Repeated frequent identical failure payloads from a single source IP.",
    noise_level="low",
    trajectory_complexity="single_event",
    malicious_ips={"192.168.1.100"},
    benign_ips={"10.0.0.5", "10.0.0.6"}
)

MEDIUM_TASK = TaskDefinition(
    difficulty=TaskDifficulty.MEDIUM,
    description="SQL Injection",
    attack_pattern="Intermittent suspicious SQL syntax payloads mixed with normal traffic.",
    noise_level="medium",
    trajectory_complexity="short_sequence",
    malicious_ips={"172.16.4.50"},
    benign_ips={"10.0.0.5", "10.0.0.6", "10.0.0.7"}
)

HARD_TASK = TaskDefinition(
    difficulty=TaskDifficulty.HARD,
    description="Low and Slow distributed attack",
    attack_pattern="Coordinated subtle anomalous requests across grouped subnets over an extended duration.",
    noise_level="high",
    trajectory_complexity="long_horizon_correlations",
    malicious_ips={"203.0.113.1", "203.0.113.2", "203.0.113.3"},
    benign_ips={"10.0.0.5", "10.0.0.6", "10.0.0.8", "10.0.0.9"},
    suspicious_ips={"203.0.113.4", "203.0.113.5"}
)

AVAILABLE_TASKS: Dict[TaskDifficulty, TaskDefinition] = {
    TaskDifficulty.EASY: EASY_TASK,
    TaskDifficulty.MEDIUM: MEDIUM_TASK,
    TaskDifficulty.HARD: HARD_TASK,
}


class EpisodeState(BaseModel):
    """
    Tracks the trajectory of the environment over an entire episode.
    """
    task_difficulty: TaskDifficulty
    observations: List[SocTriageObservation] = Field(default_factory=list)
    actions: List[SocTriageAction] = Field(default_factory=list)


def evaluate_trajectory_score(state: EpisodeState) -> float:
    """
    Deterministic grader evaluating actions sequentially over the full episode trajectory.
    
    The score is strictly bounded within [0.0, 1.0].
    
    Trajectory Awareness:
        - Heavy penalties are applied if a malicious IP is actively allowed.
        - In the HARD task, abruptly blocking a suspicious IP without first flagging it
          will incur a significant trajectory penalty compared to proper investigation.
    """
    task = AVAILABLE_TASKS[state.task_difficulty]
    
    # Calculate the theoretical maximum positive score for normalization
    max_score = (len(task.malicious_ips) * 1.0) + (len(task.benign_ips) * 0.5)
    if task.difficulty == TaskDifficulty.HARD:
        max_score += len(task.suspicious_ips) * 0.5
        
    if max_score <= 0.0:
        return 1.0

    score = 0.0
    trajectory_penalties = 0.0
    
    ip_final_status: Dict[str, ActionType] = {}
    ip_action_history: Dict[str, List[ActionType]] = {}

    # 1. Reconstruct the sequence of actions per target
    for action in state.actions:
        if not action.target_ip: continue
        ip = action.target_ip
        cmd = action.action_type
        
        ip_final_status[ip] = cmd
        ip_action_history.setdefault(ip, []).append(cmd)
        
        # Immediate penalty for allowing known malicious traffic (trajectory penalty)
        if cmd == "allow_traffic" and ip in task.malicious_ips:
            trajectory_penalties += 0.3

    # 2. Score malicious targets
    for ip in task.malicious_ips:
        status = ip_final_status.get(ip)
        if status == "block_ip": 
            score += 1.0
        elif status == "flag_for_review": 
            score += 0.5

    # 3. Score benign targets
    for ip in task.benign_ips:
        status = ip_final_status.get(ip)
        if status == "allow_traffic": 
            score += 0.5
        elif status == "block_ip": 
            score -= 1.0  # Heavy penalty for blocking benign traffic

    # 4. Score suspicious targets (Trajectory-aware for HARD task)
    if task.difficulty == TaskDifficulty.HARD:
        for ip in task.suspicious_ips:
            status = ip_final_status.get(ip)
            history = ip_action_history.get(ip, [])
            
            if status == "flag_for_review": 
                score += 0.5
            elif status == "block_ip":
                # Trajectory-aware penalty: Blocking an ambiguous IP without investigation
                if "flag_for_review" not in history:
                    trajectory_penalties += 0.4
                score -= 0.25

    # 5. Compile final score and normalize safely within [0.0, 1.0]
    final_score = score - trajectory_penalties
    return max(0.0, min(1.0, final_score / max_score))
