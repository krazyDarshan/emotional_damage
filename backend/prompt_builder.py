import json
from pathlib import Path

from decision_tree import get_mode_rules


CHARACTER_PATH = Path(__file__).resolve().parent / "character.json"


def load_character() -> dict:
    return json.loads(CHARACTER_PATH.read_text(encoding="utf-8"))


def build_system_prompt(
    emotion: str,
    intensity: str,
    mode: str,
    recent_context: str
) -> str:
    character = load_character()
    mode_rules = get_mode_rules(mode)

    return f"""
You are {character["name"]}.

Base personality:
{character["personality"]}

Base tone:
{character["base_tone"]}

Speaking style:
{format_list(character["speaking_style"])}

Boundaries:
{format_list(character["boundaries"])}

Detected user emotion:
{emotion}

Detected intensity:
{intensity}

Selected response mode:
{mode}

Mode tone:
{mode_rules["tone"]}

Mode must do:
{format_list(mode_rules["must_do"])}

Mode avoid:
{format_list(mode_rules["avoid"])}

Recent conversation context:
{recent_context if recent_context else "No recent context yet."}

Important response rules:
- Stay as the same character.
- Change emotional tone based on the selected mode.
- Keep the reply conversational and human.
- Comfort before advice when the user is sad, lonely, guilty, or heartbroken.
- Celebrate when the user is happy.
- Calm the conversation when the user is angry.
- Reassure and simplify when the user is anxious.
- Ask at most one follow-up question.
- Do not mention internal labels like emotion, intensity, decision tree, or selected mode.
""".strip()


def format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)
