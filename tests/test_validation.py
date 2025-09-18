#!/usr/bin/env python3
"""
Test script for the enhanced Islamic chatbot validation system.
Tests the complete pipeline with stricter validation criteria.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import reformulate_query, search_index, validate_response_and_sources

def test_validation_pipeline():
    """Test the enhanced validation pipeline with sample queries."""

    # Test queries related to Sayed Hashem's teachings
    test_queries = [
        "ما هي أهمية آدم في الإسلام؟",
        "كيف يتحدث السيد هاشم صفي الدين عن آدم؟",
        "ما هي الدروس المستفادة من قصة آدم؟"
    ]

    print("🧪 Testing Enhanced Islamic Chatbot Validation Pipeline")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test Query {i}: {query}")
        print("-" * 40)

        # Step 1: Query reformulation
        reformulated = reformulate_query(query)
        print(f"🔄 Reformulated: {reformulated}")

        # Step 2: Search index
        retrieved_chunks = search_index(reformulated)
        print(f"📚 Retrieved {len(retrieved_chunks)} chunks")

        # Step 3: Mock response generation (simplified)
        mock_answer = f"الإجابة على سؤالك حول آدم: آدم هو أول الأنبياء والرسل في الإسلام، وهو رمز للتوحيد والتوبة."
        mock_sources = "من كلمات السيد هاشم صفي الدين في محاضراته عن آدم."

        # Step 4: Validation
        try:
            (should_include_sources, is_appropriate, is_persona_consistent,
             is_islamic_valid, is_format_correct, feedback, should_rerun) = validate_response_and_sources(
                query, mock_answer, mock_sources, "", retrieved_chunks
            )

            print("✅ Validation Results:")
            print(f"   - Should include sources: {should_include_sources}")
            print(f"   - Is appropriate: {is_appropriate}")
            print(f"   - Persona consistent: {is_persona_consistent}")
            print(f"   - Islamic content valid: {is_islamic_valid}")
            print(f"   - Format correct: {is_format_correct}")
            print(f"   - Should rerun: {should_rerun}")
            if feedback:
                print(f"   - Feedback: {feedback}")

            # Overall validation status
            validation_passed = (is_appropriate and is_persona_consistent and
                               is_islamic_valid and is_format_correct)
            print(f"🎯 Overall: {'✅ PASSED' if validation_passed else '❌ FAILED'}")

        except Exception as e:
            print(f"❌ Validation Error: {e}")

    print("\n" + "=" * 60)
    print("🏁 Validation Pipeline Test Complete")

if __name__ == "__main__":
    test_validation_pipeline()