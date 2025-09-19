# utils/diacritizer.py
import re
import json
from typing import Tuple

# Unicode ranges for Arabic diacritics
DIACRITICS_RE = re.compile(r'[\u064B-\u0652\u0670\u0653]')

# helper: load persona lexicon if exists
def load_persona_lexicon(path='persona_lexicon.json'):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

PERSONA_LEXICON = load_persona_lexicon()

# Primary: try camel_tools -> mishkal -> fallback
def apply_diacritics(text: str) -> str:
    # Try Camel Tools (preferred)
    try:
        from camel_tools.diacritize import Diacritizer
        d = Diacritizer.pretrained()
        out = d.diacritize(text)
        out = apply_persona_lexicon(out)
        return out
    except Exception:
        pass

    # Try Mishkal
    try:
        import mishkal
        mish = mishkal.Mishkal()
        out = mish.tashkeel(text)
        out = apply_persona_lexicon(out)
        return out
    except Exception:
        pass

    # Fallback: apply lexicon replacements only (best-effort)
    return apply_persona_lexicon(text)

def apply_persona_lexicon(text: str, lexicon: dict = PERSONA_LEXICON) -> str:
    # Replace longest keys first to avoid partial matches
    for key in sorted(lexicon.keys(), key=lambda k: -len(k)):
        text = text.replace(key, lexicon[key])
    return text

def compute_diacritics_coverage(text: str) -> Tuple[float, list]:
    # words = sequences of Arabic letters (keep diacritics attached)
    words = re.findall(r'[\u0600-\u06FF]+(?:[\u064B-\u0652\u0670\u0653]*)', text)
    if not words:
        return 1.0, []
    flags = []
    for w in words:
        flags.append(1 if DIACRITICS_RE.search(w) else 0)
    coverage = sum(flags) / len(flags)
    return coverage, list(zip(words, flags))