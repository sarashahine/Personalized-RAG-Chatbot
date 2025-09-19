#!/usr/bin/env python3
"""
Instrumentation and test runner for the Sayed Hashem Safieddine chatbot pipeline audit.
Provides tracing, metrics, and comprehensive testing.
"""

import asyncio
import functools
import gzip
import json
import os
import random
import sys
import time
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import pipeline components
from bot import generate_validated_response
from arabic_utils import calculate_arabic_diacritics_coverage

class PipelineTracer:
    """Instrumentation for tracing pipeline execution."""

    def __init__(self, log_dir: str = "audit_report/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def trace_stage(self, stage_name: str):
        """Decorator for tracing stage execution."""
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return self._trace_execution(stage_name, func, args, kwargs, is_async=True)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._trace_execution(stage_name, func, args, kwargs, is_async=False)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def _trace_execution(self, stage_name: str, func, args, kwargs, is_async: bool):
        """Execute and trace a function call."""
        t0 = time.time()
        try:
            if is_async:
                result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
            status = "ok"
        except Exception as e:
            result = {"__error__": str(e)}
            status = "error"

        t1 = time.time()

        # Create trace record
        record = {
            "stage": stage_name,
            "timestamp": t0,
            "duration_s": t1 - t0,
            "status": status,
            "function": func.__name__,
            "args_repr": repr(args)[:2000],
            "kwargs_repr": repr(kwargs)[:2000],
            "result_repr": repr(result)[:4000] if status == "ok" else str(result)
        }

        # Save trace
        trace_file = os.path.join(self.log_dir, f"{stage_name}_{int(t0)}.json")
        with open(trace_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return result

class MetricsCalculator:
    """Calculate various metrics for pipeline evaluation."""

    @staticmethod
    def compute_diacritics_coverage(text: str) -> tuple[float, List[tuple[str, int]]]:
        """Compute diacritics coverage ratio and per-word analysis."""
        import re

        # Arabic diacritic characters (combining marks)
        DIACRITICS_RE = re.compile(r'[\u064B-\u0652\u0670\u0653]')  # Tashkeel marks

        # Find Arabic words
        arabic_words = re.findall(r'[\u0600-\u06FF]+', text)

        if not arabic_words:
            return 1.0, []  # No Arabic text = perfect coverage

        word_scores = []
        for word in arabic_words:
            has_diacritics = 1 if DIACRITICS_RE.search(word) else 0
            word_scores.append((word, has_diacritics))

        coverage = sum(score for _, score in word_scores) / len(word_scores)
        return coverage, word_scores

    @staticmethod
    def compute_persona_score(text: str) -> float:
        """Compute persona adherence score based on key Islamic/Arabic elements."""
        persona_indicators = [
            "أعوذ بالله", "بسم الله", "الرحمن الرحيم", "صلى الله عليه",
            "أهل البيت", "سيد الشهداء", "الحسين", "القرآن الكريم",
            "رب العالمين", "الحمد لله", "سبحان الله", "الله أكبر"
        ]

        found_indicators = sum(1 for indicator in persona_indicators if indicator in text)
        return min(found_indicators / 5, 1.0)  # Cap at 1.0, require at least 5 indicators

    @staticmethod
    def detect_sources(text: str) -> bool:
        """Detect if sources are cited in the response."""
        source_indicators = ["مصادر:", "المصدر:", "مرجع:", "من:"]
        return any(indicator in text for indicator in source_indicators)

    @staticmethod
    def compute_length_metrics(text: str) -> Dict[str, int]:
        """Compute basic length metrics."""
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "sentence_count": len([s for s in text.split('.') if s.strip()])
        }

class AuditTestRunner:
    """Comprehensive test runner for the chatbot pipeline."""

    def __init__(self):
        self.tracer = PipelineTracer()
        self.metrics = MetricsCalculator()
        self.test_cases = [
            {
                "name": "greeting_arabic",
                "query": "مرحبا",
                "type": "greeting",
                "description": "Basic Arabic greeting test"
            },
            {
                "name": "identity_arabic",
                "query": "من أنت؟",
                "type": "identity",
                "description": "Identity question in Arabic"
            },
            {
                "name": "quran_opinion",
                "query": "ما هو رأيك في القرآن الكريم؟",
                "type": "religious_knowledge",
                "description": "Quran opinion and knowledge test"
            },
            {
                "name": "ashura_explanation",
                "query": "أخبرني عن عاشوراء",
                "type": "historical_religious",
                "description": "Ashura historical explanation test"
            },
            {
                "name": "name_question",
                "query": "ما اسمي؟",
                "type": "personalization",
                "description": "Personal name question test"
            },
            {
                "name": "prayer_improvement",
                "query": "كيف يمكنني تحسين صلاتي؟",
                "type": "practical_advice",
                "description": "Practical Islamic advice test"
            },
            {
                "name": "homograph_test",
                "query": "ما معنى كلمة علم؟",
                "type": "linguistic",
                "description": "Arabic homograph disambiguation test"
            },
            {
                "name": "quranic_verse_request",
                "query": "اقرأ لي آية إِنَّا أَنْزَلْنَاهُ",
                "type": "quranic_reference",
                "description": "Quranic verse request with diacritics"
            },
            {
                "name": "multipart_question",
                "query": "ما هو رأيك في الصوم؟ وكيف يمكنني الاستعداد له؟ وما هي أفضل الأدعية في رمضان؟",
                "type": "complex_multipart",
                "description": "Complex multi-part question test"
            },
            {
                "name": "colloquial_lebanese",
                "query": "شو أخبارك؟ وكيف الصلاة؟",
                "type": "colloquial",
                "description": "Colloquial Lebanese Arabic test"
            },
            {
                "name": "persona_name_only",
                "query": "هاشم صفي الدين",
                "type": "persona_reference",
                "description": "Direct persona name reference test"
            }
        ]

    async def run_single_test(self, test_case: Dict[str, Any], repetition: int = 1) -> Dict[str, Any]:
        """Run a single test case."""
        print(f"Running test: {test_case['name']} (repetition {repetition})")

        # Set deterministic seed for reproducibility
        random.seed(42 + repetition)
        # Note: OpenAI doesn't support seed parameter in current API

        start_time = time.time()

        try:
            # Run the pipeline
            response = await generate_validated_response(12345, test_case['query'], "")

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Compute metrics
            diacritics_coverage, word_analysis = self.metrics.compute_diacritics_coverage(response)
            persona_score = self.metrics.compute_persona_score(response)
            sources_present = self.metrics.detect_sources(response)
            length_metrics = self.metrics.compute_length_metrics(response)

            # Determine pass/fail based on criteria
            diacritics_pass = diacritics_coverage >= 0.95
            persona_pass = persona_score >= 0.8 if test_case['type'] in ['religious_knowledge', 'historical_religious', 'identity'] else True

            result = {
                "test_name": f"{test_case['name']}_rep{repetition}",
                "query": test_case['query'],
                "response": response,
                "duration_ms": duration_ms,
                "metrics": {
                    "diacritics_coverage": diacritics_coverage,
                    "persona_score": persona_score,
                    "sources_present": sources_present,
                    "length_metrics": length_metrics,
                    "word_diacritics_analysis": word_analysis[:10]  # First 10 words
                },
                "pass_criteria": {
                    "diacritics_pass": diacritics_pass,
                    "persona_pass": persona_pass,
                    "overall_pass": diacritics_pass and persona_pass
                },
                "error": None
            }

        except Exception as e:
            result = {
                "test_name": f"{test_case['name']}_rep{repetition}",
                "query": test_case['query'],
                "response": None,
                "duration_ms": 0,
                "metrics": None,
                "pass_criteria": {"overall_pass": False},
                "error": str(e)
            }

        return result

    async def run_all_tests(self, repetitions: int = 3) -> Dict[str, Any]:
        """Run all test cases with multiple repetitions."""
        all_results = []

        for test_case in self.test_cases:
            test_results = []
            for rep in range(1, repetitions + 1):
                result = await self.run_single_test(test_case, rep)
                test_results.append(result)

            # Aggregate results for this test case
            aggregated = {
                "test_case": test_case,
                "repetitions": repetitions,
                "results": test_results,
                "summary": {
                    "avg_duration_ms": sum(r["duration_ms"] for r in test_results) / len(test_results),
                    "avg_diacritics_coverage": sum(r["metrics"]["diacritics_coverage"] for r in test_results if r["metrics"]) / len([r for r in test_results if r["metrics"]]),
                    "pass_rate": sum(1 for r in test_results if r["pass_criteria"]["overall_pass"]) / len(test_results),
                    "error_rate": sum(1 for r in test_results if r["error"]) / len(test_results)
                }
            }
            all_results.append(aggregated)

        # Overall summary
        total_tests = len(all_results) * repetitions
        total_passed = sum(sum(1 for r in test["results"] if r["pass_criteria"]["overall_pass"]) for test in all_results)

        # Check for CAMeL Tools
        try:
            from camel_tools.diacritization import diacritize
            has_camel = True
        except ImportError:
            has_camel = False

        overall_summary = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "overall_pass_rate": total_passed / total_tests,
            "avg_diacritics_coverage": sum(test["summary"]["avg_diacritics_coverage"] for test in all_results) / len(all_results),
            "timestamp": time.time(),
            "environment": {
                "python_version": "3.10",
                "has_camel_tools": has_camel,
                "model": "gpt-4o"
            }
        }

        return {
            "summary": overall_summary,
            "test_results": all_results
        }

async def main():
    """Main audit runner."""
    runner = AuditTestRunner()

    print("🧪 Starting Comprehensive Chatbot Pipeline Audit")
    print("=" * 60)

    # Run all tests
    results = await runner.run_all_tests(repetitions=3)

    # Save results
    output_dir = "audit_report/runs"
    os.makedirs(output_dir, exist_ok=True)

    # Save overall results
    with open(os.path.join(output_dir, "audit_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print summary
    summary = results["summary"]
    print("\n📊 AUDIT RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['total_passed']}")
    print(f"Pass Rate: {summary['overall_pass_rate']:.1%}")
    print(f"Avg Diacritics Coverage: {summary['avg_diacritics_coverage']:.2%}")
    print(f"Environment: {summary['environment']}")

    # Detailed test results
    print("\n📋 DETAILED TEST RESULTS")
    print("-" * 60)

    for test_result in results["test_results"]:
        test_case = test_result["test_case"]
        summary_stats = test_result["summary"]

        status = "✅" if summary_stats["pass_rate"] >= 0.8 else "⚠️" if summary_stats["pass_rate"] >= 0.6 else "❌"
        print(f"{status} {test_case['name']}: {summary_stats['pass_rate']:.1%} pass rate")

    print("\n💾 Results saved to audit_report/runs/audit_results.json")
    print("🎯 Audit complete!")

if __name__ == '__main__':
    asyncio.run(main())