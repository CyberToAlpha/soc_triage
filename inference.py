import os
import json
import logging
from typing import Any, Dict, List

from huggingface_hub import InferenceClient

try:
    from models import SocTriageAction, SocTriageObservation
except ImportError:
    from .models import SocTriageAction, SocTriageObservation

logger = logging.getLogger(__name__)

# Fallback to an open-weights standard if explicit HF_MODEL_ID isn't set.
MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")


def get_triage_action(
    client: InferenceClient, 
    obs: SocTriageObservation, 
    history: List[str]
) -> SocTriageAction:
    """
    Submits the observation to a Hugging Face model and parses a valid SocTriageAction.
    """
    prompt = f"""You are an elite SOC Analyst. Analyze this network observation and decide the next course of action.

Available Action Types:
1. "block_ip" - Use if strictly malicious.
2. "allow_traffic" - Use if clearly benign.
3. "flag_for_review" - Use if ambiguous and requires cautious investigation first.

Recent Event History:
{chr(10).join(history) if history else 'No previous events in this episode.'}

Current Observation:
- Time: {obs.timestamp}
- IP: {obs.source_ip}
- Request: {obs.request_payload}
- Status: {obs.http_status}
- Browser: {obs.user_agent}

You must respond STRICTLY with a valid JSON object. No other text or markdown wrapping.
Schema:
{{
  "action_type": "<action_type>",
  "target_ip": "{obs.source_ip}",
  "rationale": "<reasoning>"
}}
"""
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1
        )
        
        # Open-weight models sometimes add markdown or chatter despite prompt engineering.
        raw_text = response.choices[0].message.content.strip()
        
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        action_dict = json.loads(raw_text.strip())
        
        return SocTriageAction(
            action_type=action_dict.get("action_type", "flag_for_review"),
            target_ip=action_dict.get("target_ip", obs.source_ip),
            rationale=action_dict.get("rationale", "Automated HF triage decision.")
        )
        
    except Exception as e:
        logger.warning(f"Failed to extract JSON from LLM, falling back. Error: {e}")
        return SocTriageAction(
            action_type="flag_for_review",
            target_ip=obs.source_ip,
            rationale=f"Fallback triggered due to inference error."
        )


def run_inference(env: Any) -> float:
    """
    The main evaluator loop utilized by the OpenEnv submission runner.
    """
    token = os.environ.get("HF_TOKEN")
    client = InferenceClient(model=MODEL_ID, token=token)
    
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
        action_obj = get_triage_action(client, obs, history)
        
        # Slide history window for trajectory memory
        history.append(f"[{obs.timestamp:.1f}] IP: {obs.source_ip} | Action: {action_obj.action_type}")
        if len(history) > 4:
            history.pop(0)
            
        # Depending on EnvClient implementation, pass as dict or object. 
        try:
            step_result = env.step(action_obj)
        except Exception:
            step_result = env.step(action_obj.model_dump())
            
        # Standardize result extraction depending on protocol wrapper layer
        if isinstance(step_result, dict):
            # Client over HTTP wraps payload usually
            obs = SocTriageObservation.model_validate(step_result.get("observation", step_result))
            total_reward = step_result.get("reward", 0.0)
            done = step_result.get("done", True)
        else:
            # Native environment evaluation block
            obs = step_result
            if hasattr(obs, 'reward'):
                total_reward = obs.reward
            if hasattr(obs, 'done'):
                done = obs.done
            elif hasattr(step_result, 'observation'):
                obs = step_result.observation
                total_reward = step_result.reward
                done = step_result.done
                
    return total_reward


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Direct local injection for sandbox testing
    try:
        from server.soc_triage_environment import SocTriageEnvironment
    except ImportError:
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from server.soc_triage_environment import SocTriageEnvironment
        
    print(f"🚀 Starting OpenEnv SOC Triage Inference Pipeline")
    print(f"🎯 Using Model: {MODEL_ID}")
    
    local_env = SocTriageEnvironment()
    final_score = run_inference(local_env)
    
    print(f"\\n✅ Execution Finished!")
    print(f"🏆 Final Graded Score: {final_score:.2f} / 1.0")