"""
Diagnostic script — run this INSIDE your Covenant folder (same place as main.py)
to test whether your OpenAI key + model actually work.

Run:
    python test_openai_key.py
"""

import os
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    print("OpenAI API key configured:", bool(api_key))

    if not api_key:
        print("No key was found. Copy .env.example to .env and replace the placeholder value.")
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        return 1

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        'Extract fields from this text and return ONLY JSON with keys '
                        'fields, confidence, issues. Text: "meri income 70,000 monthly"'
                    ),
                }
            ],
        )
        print("SUCCESS! Raw response from OpenAI:")
        print(response.choices[0].message.content)
        return 0
    except Exception as exc:
        print("FAILED. Here is the exact error:")
        print(repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
