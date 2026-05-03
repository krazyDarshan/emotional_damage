from dataclasses import dataclass


@dataclass
class EmotionResult:
    emotion: str
    intensity: str
    risk: str


KEYWORDS = {
    "happy": [
        "happy", "excited", "proud", "great", "amazing", "passed", "won",
        "love this", "good news", "celebrate", "finally did it"
    ],
    "sad": [
        "sad", "cry", "crying", "hurt", "depressed", "empty", "broken",
        "hopeless", "upset", "low", "tired of everything"
    ],
    "lonely": [
        "lonely", "alone", "nobody cares", "ignored", "unseen",
        "left out", "abandoned"
    ],
    "heartbroken": [
        "breakup", "broke up", "heartbroken", "miss her", "miss him",
        "cheated", "left me", "relationship ended"
    ],
    "angry": [
        "angry", "mad", "furious", "hate", "annoyed", "rage",
        "irritated", "revenge"
    ],
    "anxious": [
        "anxious", "anxiety", "panic", "scared", "worried", "nervous",
        "overthinking", "can't breathe", "stressed"
    ],
    "romantic": [
        "crush", "romantic", "love her", "love him", "date", "flirt",
        "relationship", "feelings for"
    ],
    "confused": [
        "confused", "don't understand", "what should i do", "lost",
        "unclear", "mixed signals", "help me understand"
    ],
    "guilty": [
        "guilty", "my fault", "regret", "ashamed", "i messed up",
        "sorry for what i did"
    ]
}

CRISIS_KEYWORDS = [
    "kill myself",
    "end my life",
    "suicide",
    "self harm",
    "hurt myself",
    "hurt someone",
    "i want to die",
    "i don't want to live",
    "no reason to live"
]

HIGH_INTENSITY_WORDS = [
    "very", "really", "so much", "extremely", "can't", "cannot",
    "always", "never", "unbearable", "completely", "totally"
]


def detect_emotion(text: str) -> EmotionResult:
    normalized = text.lower().strip()

    if any(keyword in normalized for keyword in CRISIS_KEYWORDS):
        return EmotionResult(emotion="crisis", intensity="high", risk="crisis")

    scores = {}
    for emotion, keywords in KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in normalized)
        if score:
            scores[emotion] = score

    if not scores:
        return EmotionResult(emotion="neutral", intensity="low", risk="normal")

    emotion = max(scores, key=scores.get)
    intensity = "high" if any(word in normalized for word in HIGH_INTENSITY_WORDS) else "medium"

    return EmotionResult(emotion=emotion, intensity=intensity, risk="normal")
