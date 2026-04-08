import os
import json
import logging
from typing import Any, Dict, List
from openai import OpenAI
try:
    from models import SocTriageAction, SocTriageObservation
except ImportError:
    from .models import SocTriageAction, SocTriageObservation
logger = logging.getLogger(__name__)
# Strictly mandated environment variables from the Hackathon rules
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
def get_action_from_llm(
    client: OpenAI, observation_data: Dict[str, Any], history: List[str]
) -> SocTriageAction:
    """
    Queries an LLM using the strictly required OpenAI client format to decide the next SOC action.
    """
    try:
        obs = SocTriageObservation(**observation_data)
        prompt = f"""You are an expert SOC Analyst triaging network security events.
Based on the following network log observation and recent history, decide what action to take.
Your available action types are strictly:
1. "block_ip" - Use if the activity is confidently malicious.
2. "allow_traffic" - Use if the activity is benign.
3. "flag_for_review" - Use if the activity is suspicious or ambiguous.
Recent Network History (Last 5 events):
{chr(10).join(history) if history else "No previous logs."}
Current Observation:
- Timestamp: {obs.timestamp}
- Source IP: {obs.source_ip}
- Payload: {obs.request_payload}
Respond strictly with a valid JSON object matching this schema, and nothing else (no markdown wrapping):
{{
  "action_type": "<action_type>",
  "target_ip": "{obs.source_ip}",
  "rationale": "<optional brief reason>"
}}
"""
        # Using the mandated OpenAI client structure
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        response_text = response.choices[0].message.content.strip()
        # Clean formatting
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        action_dict = json.loads(response_text.strip())
        return SocTriageAction(
            action_type=action_dict.get("action_type", "flag_for_review"),
            target_ip=action_dict.get("target_ip", obs.source_ip),
            rationale=action_dict.get("rationale"),
        )
    except Exception as e:
        logger.warning(f"Failed to extract JSON from LLM: {e}")
        return SocTriageAction(
            action_type="flag_for_review",
            target_ip=observation_data.get("source_ip", "unknown")
            if isinstance(observation_data, dict)
            else "unknown",
            rationale=f"Fallback due to inference error: {str(e)}",
        )
def run_inference(env: Any) -> float:
    """
    The main evaluator loop utilizing the OpenAI client to HuggingFace Spaces.
    """
    # Create the client explicitly respecting standard OpenAI kwargs mapped to HF Base URLs
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
        or "hf_fake_token_for_validation",  # HF token used as API key in OpenAI API compliant standard
    )
    # Initialize the trajectory
    obs_raw = env.reset()
    # Check if observation comes as a dict (HTTP Client) or object (Local testing)
    if isinstance(obs_raw, dict):
        obs = SocTriageObservation.model_validate(obs_raw.get("observation", obs_raw))
    else:
        obs = obs_raw
    done = False
    total_reward = 0.0
    history: List[str] = []
    while not done:
        # Determine the target action
        action_obj = get_action_from_llm(client, obs.model_dump(), history)
        # Slide history window for trajectory memory
        history.append(
            f"[{obs.timestamp:.1f}] IP: {obs.source_ip} | Action: {action_obj.action_type}"
        )
        if len(history) > 4:
            history.pop(0)
        # Step the environment via OpenEnv standard (try sending object, fallback to dict)
        try:
            step_result = env.step(action_obj)
        except Exception:
            step_result = env.step(action_obj.model_dump())
        # Standardize result extraction depending on protocol wrapper layer
        if isinstance(step_result, dict):
            # Client over HTTP wraps payload usually
            obs = SocTriageObservation.model_validate(
                step_result.get("observation", step_result)
            )
            total_reward = step_result.get("reward", 0.0)
            done = step_result.get("done", True)
        else:
            # Native environment evaluation block
            obs = step_result
            if hasattr(obs, "reward"):
                total_reward = obs.reward
            if hasattr(obs, "done"):
                done = obs.done
            elif hasattr(step_result, "observation"):
                obs = step_result.observation
                total_reward = step_result.reward
                done = step_result.done
    return total_reward
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Direct local injection for sandbox testing
    try:
        from server.soc_triage_environment import SocTriageEnvironment
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from server.soc_triage_environment import SocTriageEnvironment
    logger.info("🚀 Starting OpenEnv SOC Triage Inference Pipeline")
    logger.info(f"🎯 Using Model via OpenAI API Format: {MODEL_NAME}")
    logger.info(f"🔗 API Endpoint: {API_BASE_URL}")
    local_env = SocTriageEnvironment()
    final_score = run_inference(local_env)
    logger.info("✅ Execution Finished!")
    logger.info(f"🏆 Final Graded Score: {final_score:.2f} / 1.0")
