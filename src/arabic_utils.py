# arabic_utils.py
"""
Arabic language processing utilities for the Sayed Hashem Safieddine chatbot.
Includes diacritization, text processing, and persona-specific formatting.
"""

import json
import os
import re
from typing import Dict, List, Optional

try:
    from camel_tools.diacritization import diacritize
    CAMEL_AVAILABLE = True
    print("✅ CAMeL Tools available for diacritization")
except ImportError as e:
    CAMEL_AVAILABLE = False
    print(f"⚠️  CAMeL Tools not available. Diacritization will be limited. Error: {e}")
    diacritize = None

class ArabicDiacritizer:
    """Arabic text diacritization and persona-specific formatting."""

    def __init__(self, lexicon_path: str = None):
        self.lexicon = {}
        if lexicon_path and os.path.exists(lexicon_path):
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                self.lexicon = json.load(f)

    def apply_lexicon(self, text: str) -> str:
        """Apply persona lexicon mappings to override automatic diacritization."""
        result = text
        # Sort by length (longest first) to handle overlapping matches
        sorted_mappings = sorted(self.lexicon.items(), key=lambda x: len(x[0]), reverse=True)

        for plain, diacritized in sorted_mappings:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(plain) + r'\b'
            result = re.sub(pattern, diacritized, result)

        return result

    def diacritize_text(self, text: str) -> str:
        """Apply full diacritization pipeline: automatic + lexicon override."""
        if not CAMEL_AVAILABLE:
            # Fallback: just apply lexicon
            return self.apply_lexicon(text)

        try:
            # Step 1: Automatic diacritization using CAMeL Tools
            auto_diacritized = diacritize(text)[0]  # diacritize returns a list

            # Step 2: Apply persona lexicon to override specific terms
            final_text = self.apply_lexicon(auto_diacritized)

            return final_text

        except Exception as e:
            print(f"Diacritization error: {e}")
            # Fallback to lexicon-only
            return self.apply_lexicon(text)

    def calculate_diacritics_coverage(self, text: str) -> float:
        """Calculate the ratio of words with proper Arabic diacritics."""
        if not text.strip():
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        # Arabic diacritic characters
        diacritics = {'َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ', 'ٰ', 'ٓ', 'ٔ', 'ٕ', 'ٖ', 'ٗ'}

        words_with_diacritics = 0
        total_arabic_words = 0

        for word in words:
            # Check if word contains Arabic characters
            if any('\u0600' <= char <= '\u06FF' for char in word):
                total_arabic_words += 1
                # Check if word has diacritics
                if any(char in diacritics for char in word):
                    words_with_diacritics += 1

        return words_with_diacritics / total_arabic_words if total_arabic_words > 0 else 0.0

    def ensure_full_diacritization(self, text: str, min_coverage: float = 0.95) -> str:
        """Ensure text meets minimum diacritics coverage, re-diacritizing if needed."""
        current_coverage = self.calculate_diacritics_coverage(text)

        if current_coverage >= min_coverage:
            return text

        # Re-diacritize if coverage is insufficient
        print(f"Low diacritics coverage ({current_coverage:.2f}), re-diacritizing...")
        return self.diacritize_text(text)

# Global diacritizer instance
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON_PATH = os.path.join(PROJECT_ROOT, "config", "persona_lexicon.json")
diacritizer = ArabicDiacritizer(LEXICON_PATH)

def diacritize_arabic_text(text: str) -> str:
    """Convenience function for diacritizing Arabic text."""
    return diacritizer.diacritize_text(text)

def calculate_arabic_diacritics_coverage(text: str) -> float:
    """Calculate diacritics coverage ratio."""
    return diacritizer.calculate_diacritics_coverage(text)

def ensure_arabic_diacritization(text: str, min_coverage: float = 0.95) -> str:
    """Ensure text has sufficient Arabic diacritization."""
    return diacritizer.ensure_full_diacritization(text, min_coverage)