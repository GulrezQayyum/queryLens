import os
import json
from groq import Groq

from config import GROQ_MODEL

_client = None

def _get_client():
    global _client
    if _client is None:
                _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client
        
def _ask(prompt: str, system: str = "You are a concise, precise assistant.") -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()
    
def hyde(query: str) -> str:
    prompt = (
        f"Write a short, plausible passage (3-4 sentences) that would answer "
        f"this question, as if it were an excerpt from a technical document. "
        f"Do not say you don't know — write your best guess even if unsure.\n\n"
        f"Question: {query}"
    )

    return _ask(prompt)

def multi_query(query: str, n: int = 4) -> list[str]:
    prompt = (
        f"Rewrite the following question in {n} different ways, using different "
        f"vocabulary and phrasing but preserving the original meaning. "
        f"Respond with ONLY a JSON array of {n} strings, nothing else.\n\n"
        f"Question: {query}"
    )
    
    raw = _ask(prompt, system="You respond only with valid JSON arrays of strings, no other text.")
    try:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        variants = json.loads(cleaned)
        if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
            return variants[:n]
    except (json.JSONDecodeError, ValueError):
        pass
        return [query] * n
    
def step_back(query: str) -> str:
    prompt = (
        f"Given this specific question, write a single more general or "
        f"abstract question about the broader concept it depends on. "
        f"Respond with ONLY the general question, nothing else.\n\n"
        f"Question: {query}"
    )
    return _ask(prompt)
ROUTES = ["direct", "hyde", "multi_query", "step_back"]
    
def route(query: str) -> str:
    prompt = (
        f"Classify this question into exactly one category:\n"
        f"- 'direct': short, well-specified questions using vocabulary likely "
        f"to appear directly in a technical document\n"
        f"- 'hyde': open-ended or vague questions where a hypothetical answer "
        f"would help retrieval more than the question itself\n"
        f"- 'multi_query': questions that could be phrased many different ways, "
        f"or use ambiguous/informal vocabulary\n"
        f"- 'step_back': narrow or highly specific questions that probably need "
        f"broader background context to answer well\n\n"
        f"Question: {query}\n\n"
        f"Respond with ONLY one word: direct, hyde, multi_query, or step_back."
    )
    result = _ask(prompt, system="You respond with exactly one word, no punctuation, no explanation.")
    result = result.strip().lower().replace("'", "").replace('"', "")
    return result if result in ROUTES else "direct"
    