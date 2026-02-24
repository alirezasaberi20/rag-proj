#!/usr/bin/env python3
"""
Test script for the RAG Chatbot API.

Run this script to test all API endpoints.
Requires the API server to be running.
"""

import httpx
import json
import sys

BASE_URL = "http://localhost:8000"


def print_response(name: str, response: httpx.Response):
    """Pretty print API response."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
    except Exception:
        print(f"Response: {response.text}")
    print()


def test_health():
    """Test health endpoint."""
    response = httpx.get(f"{BASE_URL}/health")
    print_response("Health Check", response)
    return response.status_code == 200


def test_ready():
    """Test readiness endpoint."""
    response = httpx.get(f"{BASE_URL}/ready")
    print_response("Readiness Check", response)
    return response.status_code == 200


def test_models():
    """Test models listing."""
    response = httpx.get(f"{BASE_URL}/models")
    print_response("List Models", response)
    return response.status_code == 200


def test_ingest():
    """Test document ingestion."""
    documents = {
        "documents": [
            {
                "content": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                "metadata": {"source": "python-intro", "topic": "programming"}
            },
            {
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
                "metadata": {"source": "ml-intro", "topic": "ai"}
            },
            {
                "content": "FastAPI is a modern, fast web framework for building APIs with Python. It's based on standard Python type hints and provides automatic API documentation. FastAPI is one of the fastest Python frameworks available.",
                "metadata": {"source": "fastapi-intro", "topic": "web"}
            },
        ]
    }

    response = httpx.post(
        f"{BASE_URL}/api/v1/documents/ingest",
        json=documents,
        timeout=60.0,
    )
    print_response("Ingest Documents", response)
    return response.status_code == 200


def test_stats():
    """Test document stats."""
    response = httpx.get(f"{BASE_URL}/api/v1/documents/stats")
    print_response("Document Stats", response)
    return response.status_code == 200


def test_chat_rag():
    """Test RAG chat."""
    request = {
        "message": "What is Python and who created it?",
        "include_sources": True
    }

    response = httpx.post(
        f"{BASE_URL}/api/v1/chat",
        json=request,
        timeout=120.0,
    )
    print_response("RAG Chat", response)
    return response.status_code == 200


def test_chat_direct():
    """Test direct chat (no RAG)."""
    request = {
        "message": "Say hello in 3 different languages."
    }

    response = httpx.post(
        f"{BASE_URL}/api/v1/chat/direct",
        json=request,
        timeout=120.0,
    )
    print_response("Direct Chat", response)
    return response.status_code == 200


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RAG CHATBOT API - TEST SUITE")
    print("="*60)

    tests = [
        ("Health Check", test_health),
        ("Readiness Check", test_ready),
        ("List Models", test_models),
        ("Ingest Documents", test_ingest),
        ("Document Stats", test_stats),
        ("RAG Chat", test_chat_rag),
        ("Direct Chat", test_chat_direct),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except httpx.ConnectError:
            print(f"\nERROR: Cannot connect to API at {BASE_URL}")
            print("Make sure the server is running: uvicorn app.main:app --reload")
            sys.exit(1)
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results.append((name, False))

    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
