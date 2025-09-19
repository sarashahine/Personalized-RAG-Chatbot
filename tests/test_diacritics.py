#!/usr/bin/env python3
"""
Unit tests for Arabic diacritization functionality in the Sayed Hashem Safieddine chatbot.
Tests diacritics coverage, persona lexicon application, and pipeline integration.
"""

import unittest
import asyncio
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arabic_utils import (
    ArabicDiacritizer,
    diacritize_arabic_text,
    calculate_arabic_diacritics_coverage,
    ensure_arabic_diacritization
)
from bot import generate_validated_response

class TestArabicDiacritization(unittest.TestCase):
    """Test cases for Arabic diacritization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.diacritizer = ArabicDiacritizer()
        # Test lexicon for common Islamic terms
        self.test_lexicon = {
            "الله": "ٱللَّهِ",
            "محمد": "مُحَمَّدٍ",
            "السلام عليكم": "ٱلسَّلَامُ عَلَيْكُمْ"
        }

    def test_lexicon_application(self):
        """Test that persona lexicon correctly overrides automatic diacritization."""
        text = "الله محمد السلام عليكم"
        expected = "ٱللَّهِ مُحَمَّدٍ ٱلسَّلَامُ عَلَيْكُمْ"

        # Create diacritizer with test lexicon
        test_diacritizer = ArabicDiacritizer()
        test_diacritizer.lexicon = self.test_lexicon

        result = test_diacritizer.apply_lexicon(text)
        # Check that lexicon terms are applied
        for term in self.test_lexicon.values():
            self.assertIn(term, result)

    def test_diacritics_coverage_calculation(self):
        """Test calculation of diacritics coverage ratio."""
        # Fully diacritized text
        fully_diacritized = "ٱلسَّلَامُ عَلَيْكُمْ وَرَحْمَةُ ٱللَّهِ وَبَرَكَاتُهُ"
        coverage = calculate_arabic_diacritics_coverage(fully_diacritized)
        self.assertGreaterEqual(coverage, 0.95)

        # Partially diacritized text
        partial = "السلام عليكم ورحمة الله وبركاته"
        coverage = calculate_arabic_diacritics_coverage(partial)
        self.assertLess(coverage, 0.5)

        # No Arabic text
        no_arabic = "Hello world"
        coverage = calculate_arabic_diacritics_coverage(no_arabic)
        self.assertEqual(coverage, 0.0)

    def test_ensure_diacritization(self):
        """Test that ensure_arabic_diacritization meets minimum coverage."""
        partial_text = "السلام عليكم ورحمة الله"
        result = ensure_arabic_diacritization(partial_text, min_coverage=0.8)

        coverage = calculate_arabic_diacritics_coverage(result)
        self.assertGreaterEqual(coverage, 0.8)

    def test_greeting_diacritization(self):
        """Test diacritization of common Islamic greetings."""
        greetings = [
            "السلام عليكم ورحمة الله وبركاته",
            "أعوذ بالله من الشيطان الرجيم",
            "بسم الله الرحمن الرحيم"
        ]

        for greeting in greetings:
            with self.subTest(greeting=greeting):
                diacritized = diacritize_arabic_text(greeting)
                coverage = calculate_arabic_diacritics_coverage(diacritized)
                self.assertGreaterEqual(coverage, 0.9,
                    f"Greeting '{greeting}' has insufficient diacritization: {coverage}")

    def test_identity_response_diacritization(self):
        """Test diacritization of identity responses."""
        identity_text = "أنا هاشم صفي الدين خادم لأهل البيت سلام الله عليهم"
        diacritized = diacritize_arabic_text(identity_text)
        coverage = calculate_arabic_diacritics_coverage(diacritized)
        self.assertGreaterEqual(coverage, 0.9)

    def test_quranic_phrase_diacritization(self):
        """Test diacritization of Quranic phrases."""
        quranic = "قل هو الله أحد الله الصمد"
        diacritized = diacritize_arabic_text(quranic)
        coverage = calculate_arabic_diacritics_coverage(diacritized)
        self.assertGreaterEqual(coverage, 0.9)

    def test_long_explanation_diacritization(self):
        """Test diacritization of longer explanatory text."""
        long_text = """
        عاشوراء يوم عظيم في تاريخ الأمة الإسلامية يحيي فيه المؤمنون ذكرى استشهاد
        سيد الشهداء الإمام الحسين بن علي سلام الله عليه في كربلاء
        """

        diacritized = diacritize_arabic_text(long_text)
        coverage = calculate_arabic_diacritics_coverage(diacritized)
        self.assertGreaterEqual(coverage, 0.85)

    @patch('bot.client')
    async def test_full_pipeline_diacritization(self, mock_client):
        """Test that the full chatbot pipeline produces fully diacritized responses."""
        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "السلام عليكم ورحمة الله وبركاته"
        mock_client.chat.completions.create.return_value = mock_response

        # Test greeting
        response = await generate_validated_response(12345, "مرحبا", "")
        coverage = calculate_arabic_diacritics_coverage(response)

        # The response should be fully diacritized
        self.assertGreaterEqual(coverage, 0.95,
            f"Full pipeline response has insufficient diacritization: {coverage}")

    def test_diacritizer_fallback(self):
        """Test that diacritizer works even when CAMeL Tools is not available."""
        # Temporarily disable CAMeL
        import arabic_utils
        original_camel = arabic_utils.CAMEL_AVAILABLE
        arabic_utils.CAMEL_AVAILABLE = False

        try:
            text = "السلام عليكم"
            result = diacritize_arabic_text(text)
            # Should still apply lexicon even without CAMeL
            self.assertIsInstance(result, str)
        finally:
            # Restore original state
            arabic_utils.CAMEL_AVAILABLE = original_camel

if __name__ == '__main__':
    # Install CAMeL Tools if not available (for testing)
    try:
        import camel_tools
    except ImportError:
        print("Installing CAMeL Tools for testing...")
        os.system("pip install camel-tools")

    unittest.main()