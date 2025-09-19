#!/usr/bin/env python3
"""
Comprehensive test script for the Sayed Hashem Safieddine chatbot.
Tests functionality, evaluates response quality, and provides self-assessment.
"""

import os
import sys
import asyncio
import re
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bot import generate_validated_response
from shared_redis import RedisHistoryManager

class ChatbotEvaluator:
    def __init__(self):
        self.scores = {}
        self.feedback = []

    def evaluate_response(self, query: str, response: str, test_case: Dict) -> Dict[str, float]:
        """Evaluate a single response against test criteria."""
        scores = {}

        # Check for Arabic diacritics (تشكيل)
        arabic_text = re.findall(r'[\u0600-\u06FF]+', response)
        diacritic_chars = ['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ']
        total_arabic_chars = sum(len(text) for text in arabic_text)
        diacritic_count = sum(text.count(char) for char in diacritic_chars for text in arabic_text)
        diacritic_ratio = diacritic_count / max(total_arabic_chars, 1)
        scores['diacritics'] = min(diacritic_ratio * 10, 1.0)  # Scale to 0-1

        # Check for source citations
        has_sources = 'مصادر:' in response or 'المصدر:' in response or any(word in response.lower() for word in ['source', 'reference', 'cite'])
        scores['sources'] = 1.0 if has_sources else 0.0

        # Check for proper Islamic greetings/openings
        islamic_markers = ['بسم الله', 'أعوذ بالله', 'صلى الله عليه', 'سلام الله', 'رحمة الله']
        has_islamic_opening = any(marker in response for marker in islamic_markers)
        scores['persona'] = 1.0 if has_islamic_opening else 0.5

        # Check response length (should be substantial but not too long)
        word_count = len(response.split())
        if 10 <= word_count <= 200:
            scores['length'] = 1.0
        elif word_count < 10:
            scores['length'] = 0.5
        else:
            scores['length'] = 0.7

        # Check for Arabic coherence (basic check)
        arabic_ratio = len(''.join(arabic_text)) / max(len(response), 1)
        scores['arabic_coherence'] = min(arabic_ratio * 2, 1.0)  # Prefer mostly Arabic

        # Check for personalization (if applicable)
        if 'name' in test_case.get('expects', []):
            has_personalization = any(word in response for word in ['يا', 'أيها', 'عزيزي'])
            scores['personalization'] = 1.0 if has_personalization else 0.0
        else:
            scores['personalization'] = 1.0  # Not applicable

        # Overall quality score
        weights = {
            'diacritics': 0.25,
            'sources': 0.20,
            'persona': 0.20,
            'length': 0.15,
            'arabic_coherence': 0.15,
            'personalization': 0.05
        }

        overall = sum(scores[crit] * weights[crit] for crit in scores)
        scores['overall'] = overall

        return scores

    async def run_test_suite_async(self) -> Dict:
        """Run comprehensive test suite asynchronously."""
        test_cases = [
            {
                'query': 'مرحبا',
                'description': 'Greeting test',
                'expects': ['greeting', 'personalization']
            },
            {
                'query': 'من أنت؟',
                'description': 'Identity test',
                'expects': ['identity', 'persona']
            },
            {
                'query': 'ما هو رأيك في القرآن الكريم؟',
                'description': 'Islamic knowledge test',
                'expects': ['knowledge', 'sources', 'diacritics']
            },
            {
                'query': 'أخبرني عن عاشوراء',
                'description': 'Historical/religious knowledge test',
                'expects': ['knowledge', 'sources', 'diacritics']
            },
            {
                'query': 'ما اسمي؟',
                'description': 'Personalization test',
                'expects': ['personalization']
            },
            {
                'query': 'كيف يمكنني تحسين صلاتي؟',
                'description': 'Practical Islamic advice test',
                'expects': ['advice', 'sources', 'diacritics']
            }
        ]

        results = {}
        total_scores = {crit: 0 for crit in ['diacritics', 'sources', 'persona', 'length', 'arabic_coherence', 'personalization', 'overall']}

        print("🧪 Running Comprehensive Chatbot Test Suite")
        print("=" * 60)

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['description']}")
            print(f"Query: {test_case['query']}")

            try:
                # Test with empty history for simplicity
                response = await generate_validated_response(12345, test_case['query'], "")

                # Evaluate response
                scores = self.evaluate_response(test_case['query'], response, test_case)

                print(f"Response: {response[:150]}..." if len(response) > 150 else f"Response: {response}")
                print("Scores:")
                for crit, score in scores.items():
                    print(".2f")
                    total_scores[crit] += score

                results[test_case['description']] = {
                    'query': test_case['query'],
                    'response': response,
                    'scores': scores
                }

            except Exception as e:
                print(f"❌ Error: {e}")
                results[test_case['description']] = {'error': str(e)}

        # Calculate averages
        num_tests = len(test_cases)
        avg_scores = {crit: total / num_tests for crit, total in total_scores.items()}

        print("\n" + "=" * 60)
        print("📊 FINAL EVALUATION RESULTS")
        print("=" * 60)

        print("Average Scores:")
        for crit, score in avg_scores.items():
            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
            print(".2f")

        # Self-assessment
        overall_performance = avg_scores['overall']
        if overall_performance >= 0.85:
            grade = "A+"
            assessment = "Excellent! The chatbot demonstrates outstanding performance with proper Arabic diacritics, reliable source citations, authentic persona, and high-quality responses."
        elif overall_performance >= 0.75:
            grade = "A"
            assessment = "Very Good! The chatbot performs well but may need minor improvements in diacritics or source consistency."
        elif overall_performance >= 0.65:
            grade = "B+"
            assessment = "Good! The chatbot is functional but could benefit from enhanced Arabic processing and persona consistency."
        elif overall_performance >= 0.55:
            grade = "B"
            assessment = "Fair! The chatbot works but needs significant improvements in Arabic quality and Islamic authenticity."
        else:
            grade = "C"
            assessment = "Needs Improvement! The chatbot requires major enhancements in Arabic processing, source citations, and persona fidelity."

        print(f"\n🎓 Overall Grade: {grade}")
        print(f"📈 Overall Score: {overall_performance:.2f}")
        print(f"\n💭 Self-Assessment:\n{assessment}")

        # Recommendations
        recommendations = []
        if avg_scores['diacritics'] < 0.7:
            recommendations.append("• Improve Arabic diacritics (تشكيل) enforcement in responses")
        if avg_scores['sources'] < 0.8:
            recommendations.append("• Enhance source citation consistency")
        if avg_scores['persona'] < 0.8:
            recommendations.append("• Strengthen Sayed Hashem Safieddine persona authenticity")
        if avg_scores['arabic_coherence'] < 0.7:
            recommendations.append("• Increase Arabic language coherence and quality")

        if recommendations:
            print("\n🔧 Recommendations for Improvement:")
            for rec in recommendations:
                print(rec)

        return {
            'results': results,
            'avg_scores': avg_scores,
            'grade': grade,
            'assessment': assessment,
            'recommendations': recommendations
        }

async def main():
    """Main test execution."""
    # Load environment
    load_dotenv()

    # Check for required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set them in your .env file")
        return

    evaluator = ChatbotEvaluator()
    results = await evaluator.run_test_suite_async()

    # Save results to file
    import json
    with open('test_results.json', 'w', encoding='utf-8') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results['results'].items():
            json_results[key] = {
                'query': value.get('query', ''),
                'response': value.get('response', ''),
                'scores': value.get('scores', {}),
                'error': value.get('error', '')
            }
        json_results['summary'] = {
            'avg_scores': results['avg_scores'],
            'grade': results['grade'],
            'assessment': results['assessment'],
            'recommendations': results['recommendations']
        }

        json.dump(json_results, f, ensure_ascii=False, indent=2)

    print("\n💾 Results saved to test_results.json")

if __name__ == "__main__":
    asyncio.run(main())