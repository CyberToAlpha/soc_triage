import time
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SocTriageAction, SocTriageObservation

class SocTriageEnv(EnvClient[SocTriageAction, SocTriageObservation, State]):
    """Client for the Soc Triage Environment."""

    def _step_payload(self, action: SocTriageAction) -> Dict:
        """Convert SocTriageAction to JSON payload."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[SocTriageObservation]:
        """Parse server response."""
        obs_data = payload.get("observation", {})
        
        observation = SocTriageObservation(
            timestamp=obs_data.get("timestamp", time.time()),
            source_ip=obs_data.get("source_ip", "10.0.0.1"),
            request_payload=obs_data.get("request_payload", ""),
            http_status=obs_data.get("http_status", 200),
            user_agent=obs_data.get("user_agent", "unknown"),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {})
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server state."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
