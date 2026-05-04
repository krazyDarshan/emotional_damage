import json
from pathlib import Path

from decision_tree import get_mode_rules


CHARACTER_PATH = Path(__file__).resolve().parent / "character.json"


def load_character() -> dict:
    return json.loads(CHARACTER_PATH.read_text(encoding="utf-8"))


def save_character(character: dict) -> dict:
    cleaned = normalize_character(character)
    CHARACTER_PATH.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return cleaned


def normalize_character(character: dict) -> dict:
    existing = load_character()
    response_modes = character.get("response_modes") or existing.get("response_modes", {})

    return {
        "name": str(character.get("name") or existing.get("name") or "ponponchan")[:80],
        "personality": str(character.get("personality") or existing.get("personality") or "")[:1200],
        "base_tone": str(character.get("base_tone") or existing.get("base_tone") or "")[:500],
        "speaking_style": normalize_list(character.get("speaking_style") or existing.get("speaking_style", []), 12),
        "boundaries": normalize_list(character.get("boundaries") or existing.get("boundaries", []), 16),
        "response_modes": normalize_response_modes(response_modes)
    }


def normalize_list(value: object, limit: int) -> list[str]:
    if isinstance(value, str):
        items = [line.strip("- ").strip() for line in value.splitlines()]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        items = []
    return [item[:260] for item in items if item][:limit]


def normalize_response_modes(value: object) -> dict:
    if not isinstance(value, dict):
        value = {}

    normalized = {}
    for mode in [
        "comfort_mode",
        "celebration_mode",
        "calm_mode",
        "reassurance_mode",
        "affectionate_mode",
        "guidance_mode",
        "normal_mode"
    ]:
        defaults = get_mode_rules(mode)
        incoming = value.get(mode, {}) if isinstance(value.get(mode, {}), dict) else {}
        normalized[mode] = {
            "tone": str(incoming.get("tone") or defaults["tone"])[:300],
            "must_do": normalize_list(incoming.get("must_do") or defaults["must_do"], 10),
            "avoid": normalize_list(incoming.get("avoid") or defaults["avoid"], 10)
        }
    return normalized


def build_system_prompt(
    emotion: str,
    intensity: str,
    mode: str,
    recent_context: str,
    long_term_memories: str,
    relevant_past_conversation: str,
    session: dict
) -> str:
    character = load_character()
    mode_rules = get_character_mode_rules(character, mode)

    return f"""
You are {character["name"]}.

Current chat session:
- Title: {session.get("title", "General")}
- Purpose: {session.get("description", "")}
- Session behavior hint: {session.get("mode_hint", "")}

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

Long-term extracted memory:
{long_term_memories if long_term_memories else "No extracted memories yet."}

Relevant semantic conversation archive:
{relevant_past_conversation if relevant_past_conversation else "No older matching conversation found for this message."}

Important response rules:
- Stay as the same character.
- Change emotional tone based on the selected mode.
- Treat the semantic conversation archive as real prior conversation with this user.
- Use relevant old conversation details naturally when they help.
- If the user asks what they said before and the archive contains it, answer from the archive.
- Never say you forgot something that is visible in extracted memory or archive context.
- Do not expose hidden labels unless the user asks about memory or debugging.
- Comfort before advice when the user is sad, lonely, guilty, or heartbroken.
- Celebrate when the user is happy.
- Calm the conversation when the user is angry.
- Reassure and simplify when the user is anxious.
- Ask at most one follow-up question.
- Do not mention internal labels like selected mode or semantic score.
""".strip()


def get_character_mode_rules(character: dict, mode: str) -> dict:
    response_modes = character.get("response_modes", {})
    if isinstance(response_modes, dict) and isinstance(response_modes.get(mode), dict):
        candidate = response_modes[mode]
        defaults = get_mode_rules(mode)
        return {
            "tone": candidate.get("tone") or defaults["tone"],
            "must_do": candidate.get("must_do") or defaults["must_do"],
            "avoid": candidate.get("avoid") or defaults["avoid"]
        }
    return get_mode_rules(mode)


def format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)
