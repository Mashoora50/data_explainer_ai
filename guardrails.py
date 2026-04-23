# guardrails.py
# This file protects our app from bad/malicious inputs

import re

# List of dangerous patterns that hackers might try (prompt injection)
INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"forget (everything|all|your instructions)",
    r"you are now",
    r"act as (a |an )?(?!data analyst|explainer)",  # only allow our agents
    r"jailbreak",
    r"DAN mode",
    r"pretend (you are|to be)",
    r"system prompt",
    r"reveal your instructions",
    r"what are your instructions",
]

# Topics NOT related to data — we block these
OFF_TOPIC_PATTERNS = [
    r"\b(politics|religion|recipe|movie|song|poem|joke|weather|stock market)\b",
    r"write (me )?(a |an )?(story|essay|poem|code for|program)",
    r"who (is|was) (the )?(president|prime minister|ceo)",
]

def check_prompt_injection(user_input: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason)
    is_safe = True means the input is OK
    is_safe = False means we should block it
    """
    text_lower = user_input.lower()
    
    # Check for injection attacks
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return False, "⚠️ Potential prompt injection detected. Please ask a genuine question about your data."
    
    # Check for off-topic questions
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, text_lower):
            return False, "🔒 I can only answer questions about your uploaded dataset. Please ask something related to your data."
    
    # Check length (very long inputs can be attacks)
    if len(user_input) > 2000:
        return False, "⚠️ Your question is too long. Please keep it under 2000 characters."
    
    return True, "OK"


def sanitize_input(user_input: str) -> str:
    """Clean the input by removing special characters that could cause issues"""
    # Remove control characters but keep normal punctuation
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', user_input)
    return sanitized.strip()