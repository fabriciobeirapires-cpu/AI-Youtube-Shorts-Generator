"""Local LLM backend — usa Groq (OpenAI-compatible) ou OpenAI direto."""
import os


def call_openai_llm(prompt: str) -> str:
    """Chat Completions backend. Usa Groq se GROQ_API_KEY estiver setada, senão OpenAI."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        model = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")
    else:
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_key:
            raise RuntimeError(
                "Defina GROQ_API_KEY (recomendado) ou OPENAI_API_KEY no .env."
            )
        client = OpenAI(api_key=openai_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""
