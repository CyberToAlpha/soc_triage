import time
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SocTriageAction, SocTriageObservation, EpisodeState, TaskDifficulty, AVAILABLE_TASKS, evaluate_trajectory_score
except ImportError:
    from models import SocTriageAction, SocTriageObservation, EpisodeState, TaskDifficulty, AVAILABLE_TASKS, evaluate_trajectory_score

class SocTriageEnvironment(Environment):
    """
    Soc Triage environment server simulator logic.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the soc_triage environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.episode_state = EpisodeState(task_difficulty=TaskDifficulty.HARD)
        self.task = AVAILABLE_TASKS[TaskDifficulty.HARD]

    def reset(self) -> SocTriageObservation:
        """Reset the environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # We can randomly pick difficulty or hardcode it for demonstration
        difficulty = random.choice([TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD])
        self.episode_state = EpisodeState(task_difficulty=difficulty)
        self.task = AVAILABLE_TASKS[difficulty]
        
        return SocTriageObservation(
            timestamp=time.time(),
            source_ip=list(self.task.malicious_ips)[0] if self.task.malicious_ips else "192.168.1.1",
            request_payload="Initial suspicious login attempt",
            http_status=401,
            user_agent="Mock Env",
            reward=0.0,
            done=False,
            metadata={"step": 0}
        )

    def step(self, action: SocTriageAction) -> SocTriageObservation:  # type: ignore[override]
        """Execute a step in the environment."""
        self._state.step_count += 1
        self.episode_state.actions.append(action)
        
        # Calculate intermediate reward based on the trajectory logic
        reward = evaluate_trajectory_score(self.episode_state)
        
        # Distribute traffic and dynamic payloads based on task difficulty
        if self.task.difficulty == TaskDifficulty.HARD:
            is_malicious = random.random() < 0.2
            malicious_payload = random.choice([
                "GET /admin/config.bak HTTP/1.1",
                "POST /api/v1/auth - payload: {'user': 'admin', 'pass': 'admin'}",
                "GET /etc/passwd HTTP/1.1"
            ])
            benign_payload = "GET /assets/style.css HTTP/1.1"
            
        elif self.task.difficulty == TaskDifficulty.MEDIUM:
            is_malicious = random.random() < 0.4
            malicious_payload = random.choice([
                "SELECT * FROM users WHERE ID=1 OR 1=1",
                "UNION SELECT username, password FROM admins--",
                "'; EXEC xp_cmdshell('ping 10.0.0.1');--"
            ])
            benign_payload = "GET /search?q=reports HTTP/1.1"
            
        else: # EASY
            is_malicious = random.random() < 0.5
            malicious_payload = "Repeated failed login attempt (user: root)"
            benign_payload = "Standard GET request /index.html"

        if is_malicious and self.task.malicious_ips:
            ip = random.choice(list(self.task.malicious_ips))
            payload = malicious_payload
        else:
            ip = random.choice(list(self.task.benign_ips)) if self.task.benign_ips else "10.0.0.1"
            payload = benign_payload

        done = self._state.step_count >= 10
        
        return SocTriageObservation(
            timestamp=time.time(),
            source_ip=ip,
            request_payload=payload,
            http_status=200,
            user_agent="Mock Env",
            reward=reward,
            done=done,
            metadata={"step": self._state.step_count, "action_taken": action.action_type}
        )

    @property
    def state(self) -> State:
        return self._state
