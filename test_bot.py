#!/usr/bin/env python3
"""
Test script for the optimized Sayed Hashem Safieddine chatbot.
Tests the simplified pipeline with caching, personalization, and Arabic diacritics.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bot import generate_validated_response
from shared_redis import RedisHistoryManager, format_history_for_prompt

async def test_bot():
    # Load environment
    load_dotenv()

    # Test user ID
    user_id = 12345

    # Test queries
    test_queries = [
        "مرحبا",
        "من أنت؟",
        "ما اسمي؟",
        "ما هو رأيك في القرآن الكريم؟",
        "أخبرني عن عاشوراء"
    ]

    print("Testing optimized chatbot pipeline...")
    print("=" * 50)

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            # Empty chat history for simplicity
            chat_history = ""
            response = await generate_validated_response(user_id, query, chat_history)
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(test_bot())