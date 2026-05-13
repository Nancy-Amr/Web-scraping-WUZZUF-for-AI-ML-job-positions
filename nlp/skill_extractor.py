"""
nlp/skill_extractor.py — Advanced NLP skill extraction for WUZZUF job postings.

Improvements over the basic taxonomy lookup in scraper/parser.py:
  1. Alias/synonym map   — "sklearn", "wandb", "k8s" → canonical names
  2. N-gram matching     — catches multi-word skills ("Sentence Transformers")
  3. Context scoring     — counts how many times each skill appears
  4. Requirement tagging — REQUIRED / PREFERRED per skill, inferred from sentence context
  5. Unseen skill mining — extracts tech-looking phrases outside the taxonomy

Install extra dependency:
    pip install rapidfuzz
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from typing import Optional

try:
    from rapidfuzz import fuzz
    from rapidfuzz import process as fuzz_process
    _FUZZY_OK = True
except ImportError:
    _FUZZY_OK = False

from scraper.parser import SKILL_TAXONOMY, _SKILL_LOOKUP


# ── Alias / synonym map ────────────────────────────────────────────────────────
# Maps lowercase alias → canonical skill name (must exist in SKILL_TAXONOMY)
SKILL_ALIASES: dict[str, str] = {
    # Python / Languages
    "py":                    "Python",
    "golang":                "Go",
    "cpp":                   "C++",
    "c plus plus":           "C++",
    "c sharp":               "C#",
    "js":                    "JavaScript",
    "ts":                    "TypeScript",
    "bash scripting":        "Bash",
    "shell scripting":       "Bash",
    "shell":                 "Bash",

    # ML Frameworks
    "sklearn":               "Scikit-learn",
    "scikit learn":          "Scikit-learn",
    "sci-kit learn":         "Scikit-learn",
    "sk-learn":              "Scikit-learn",
    "torch":                 "PyTorch",
    "tf":                    "TensorFlow",
    "tf2":                   "TensorFlow",
    "tensorflow2":           "TensorFlow",
    "tensorflow 2":          "TensorFlow",
    "xgb":                   "XGBoost",
    "lgbm":                  "LightGBM",
    "huggingface":           "Hugging Face",
    "hf":                    "Hugging Face",

    # Generative AI
    "llms":                  "LLM",
    "large language model":  "LLM",
    "large language models": "LLM",
    "gpt-4":                 "GPT",
    "gpt4":                  "GPT",
    "gpt 4":                 "GPT",
    "chatgpt":               "GPT",
    "retrieval augmented generation": "RAG",
    "retrieval-augmented generation": "RAG",
    "generative adversarial network":  "GANs",
    "generative adversarial networks": "GANs",
    "variational autoencoder":  "VAE",
    "variational autoencoders": "VAE",
    "fine tuning":           "Fine-tuning",
    "finetuning":            "Fine-tuning",
    "prompt eng":            "Prompt Engineering",
    "rag pipeline":          "RAG",
    "rag pipelines":         "RAG",

    # NLP
    "named entity recognition": "NER",
    "tokenizer":             "Tokenization",
    "word embeddings":       "Word2Vec",
    "sentence transformer":  "Sentence Transformers",

    # MLOps & Cloud
    "weights and biases":    "Weights & Biases",
    "wandb":                 "Weights & Biases",
    "w and b":               "Weights & Biases",
    "k8s":                   "Kubernetes",
    "kube":                  "Kubernetes",
    "aws sagemaker":         "SageMaker",
    "google cloud":          "GCP",
    "google cloud platform": "GCP",
    "microsoft azure":       "Azure",
    "vertex":                "Vertex AI",
    "ci cd":                 "CI/CD",
    "cicd":                  "CI/CD",
    "continuous integration": "CI/CD",

    # Data Engineering
    "apache spark":          "Spark",
    "pyspark":               "Spark",
    "apache kafka":          "Kafka",
    "apache airflow":        "Airflow",
    "apache flink":          "Flink",
    "apache hadoop":         "Hadoop",
    "numpy":                 "NumPy",

    # Databases
    "postgres":              "PostgreSQL",
    "mongo":                 "MongoDB",
    "elastic":               "Elasticsearch",
    "elastic search":        "Elasticsearch",
    "ms sql":                "SQL Server",
    "mssql":                 "SQL Server",
    "chromadb":              "Chroma",
    "chroma db":             "Chroma",

    # Visualization
    "powerbi":               "Power BI",
    "power-bi":              "Power BI",

    # Soft Skills
    "problem-solving":       "Problem Solving",
    "team work":             "Teamwork",
    "team player":           "Teamwork",
    "communication skills":  "Communication",
    "leadership skills":     "Leadership",
    "detail oriented":       "Attention to Detail",
    "detail-oriented":       "Attention to Detail",
    "attention to details":  "Attention to Detail",
    "analytical skills":     "Analytical",
    "analytical thinking":   "Analytical",
    "research skills":       "Research",
}

# Combined lookup: covers taxonomy + aliases
_EXTENDED_LOOKUP: dict[str, tuple[str, str]] = {**_SKILL_LOOKUP}
for _alias, _canonical in SKILL_ALIASES.items():
    _canonical_lower = _canonical.lower()
    if _canonical_lower in _SKILL_LOOKUP:
        _cat = _SKILL_LOOKUP[_canonical_lower][1]
        _EXTENDED_LOOKUP[_alias] = (_canonical, _cat)

# All canonical names as lowercase strings for fuzzy matching pool
_FUZZY_POOL = list(_EXTENDED_LOOKUP.keys())


# ── Sentence-level requirement classifiers ─────────────────────────────────────
_REQUIRED_RE = re.compile(
    r"\b(must|required|essential|mandatory|need|needs|expected|"
    r"proficien|expert|strong background|solid|minimum|at least)\b",
    re.IGNORECASE,
)
_PREFERRED_RE = re.compile(
    r"\b(preferred|nice to have|bonus|plus|advantage|familiarity|"
    r"experience with|knowledge of|understanding of|exposure to|"
    r"ideally|desirable|optional|good to have|a plus)\b",
    re.IGNORECASE,
)

# Patterns that signal "X is a skill being mentioned"
_CONTEXT_PATTERNS = [
    r"(?:experience|expertise|proficiency|knowledge|skills?)\s+(?:in|with|of|using)\s+"
    r"([A-Za-z0-9][A-Za-z0-9 \+\#\.\-]{1,38}?)(?:[,;.\n]|$)",
    r"(?:proficient|skilled|experienced|familiar|expert)\s+(?:in|with)\s+"
    r"([A-Za-z0-9][A-Za-z0-9 \+\#\.\-]{1,38}?)(?:[,;.\n]|$)",
    r"(?:using|building with|working with|work with)\s+"
    r"([A-Za-z0-9][A-Za-z0-9 \+\#\.\-]{1,38}?)(?:[,;.\n]|$)",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+|\n+", text.strip())


def _req_level(sentence: str) -> str:
    if _REQUIRED_RE.search(sentence):
        return "required"
    if _PREFERRED_RE.search(sentence):
        return "preferred"
    return "neutral"


def _exact(token: str) -> Optional[tuple[str, str]]:
    return _EXTENDED_LOOKUP.get(token.lower().strip())


def _fuzzy(token: str, threshold: int) -> Optional[tuple[str, str]]:
    if not _FUZZY_OK or len(token) < 3:
        return None
    hit = fuzz_process.extractOne(token.lower(), _FUZZY_POOL, scorer=fuzz.token_sort_ratio)
    if hit and hit[1] >= threshold:
        return _EXTENDED_LOOKUP[hit[0]]
    return None


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_skills_advanced(
    text: str,
    fuzzy: bool = True,
    fuzzy_threshold: int = 88,
    mine_unseen: bool = True,
) -> dict:
    """
    Extract and enrich skills from raw job text.

    Args:
        text:             Combined description + requirements string.
        fuzzy:            Enable fuzzy matching for misspellings (requires rapidfuzz).
        fuzzy_threshold:  Minimum similarity score 0-100 (higher = stricter).
        mine_unseen:      Collect tech-looking phrases outside the taxonomy.

    Returns:
        {
          "skills_grouped": {
            "ML Frameworks": [
              {"name": "PyTorch", "count": 3, "requirement": "required"},
              ...
            ],
          },
          "skills_flat":      ["PyTorch", "Python", ...],   # deduplicated
          "skills_required":  ["Python", "PyTorch"],
          "skills_preferred": ["Docker"],
          "candidate_skills": ["LangGraph", "CrewAI"],      # unseen, unvalidated
          "match_method":     {"PyTorch": "exact", "Scikit-learn": "alias"},
        }
    """
    if not text:
        return {
            "skills_grouped": {},
            "skills_flat": [],
            "skills_required": [],
            "skills_preferred": [],
            "candidate_skills": [],
            "match_method": {},
        }

    # skill_name → {count, requirement, category, method}
    found: dict[str, dict] = {}

    _LEVEL_RANK = {"required": 2, "preferred": 1, "neutral": 0}

    def _register(canonical: str, category: str, level: str, method: str) -> None:
        if canonical not in found:
            found[canonical] = {
                "count": 0, "requirement": "neutral",
                "category": category, "method": method,
            }
        found[canonical]["count"] += 1
        if _LEVEL_RANK[level] > _LEVEL_RANK[found[canonical]["requirement"]]:
            found[canonical]["requirement"] = level

    for sent in _sentences(text):
        level = _req_level(sent)
        words = re.findall(r"[A-Za-z0-9\+\#\.\-]+", sent)

        # Try n-grams longest-first; track consumed word indices so a 2-gram
        # match doesn't accidentally swallow unrelated single-word skills.
        consumed: set[int] = set()
        for n in (4, 3, 2, 1):
            for i in range(len(words) - n + 1):
                if any(j in consumed for j in range(i, i + n)):
                    continue
                token = " ".join(words[i : i + n])
                hit = _exact(token)
                if hit:
                    _register(hit[0], hit[1], level, "alias" if token.lower() in SKILL_ALIASES else "exact")
                    consumed.update(range(i, i + n))
                    continue
                # Fuzzy only on single tokens to avoid noisy multi-word false positives
                if fuzzy and n == 1:
                    hit = _fuzzy(token, fuzzy_threshold)
                    if hit and hit[0] not in found:
                        _register(hit[0], hit[1], level, "fuzzy")

    # ── Build output ──────────────────────────────────────────────────────────
    skills_grouped: dict[str, list[dict]] = defaultdict(list)
    match_method: dict[str, str] = {}

    for canonical, meta in found.items():
        skills_grouped[meta["category"]].append({
            "name":        canonical,
            "count":       meta["count"],
            "requirement": meta["requirement"],
        })
        match_method[canonical] = meta["method"]

    for cat in skills_grouped:
        skills_grouped[cat].sort(key=lambda x: -x["count"])

    skills_flat     = [s["name"] for skills in skills_grouped.values() for s in skills]
    skills_required  = [s["name"] for skills in skills_grouped.values()
                        for s in skills if s["requirement"] == "required"]
    skills_preferred = [s["name"] for skills in skills_grouped.values()
                        for s in skills if s["requirement"] == "preferred"]

    # ── Unseen skill mining ───────────────────────────────────────────────────
    candidate_skills: list[str] = []
    if mine_unseen:
        known_lower = {s.lower() for s in skills_flat}
        seen_cands:  set[str] = set()
        for pattern in _CONTEXT_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                phrase = m.group(1).strip().rstrip(".,; ")
                pl = phrase.lower()
                if pl in known_lower or pl in seen_cands or len(phrase) < 3:
                    continue
                # Keep only if it looks like a tech term (has uppercase, digit, or dot)
                if re.search(r"[A-Z0-9.]", phrase):
                    seen_cands.add(pl)
                    candidate_skills.append(phrase)
        candidate_skills = candidate_skills[:20]

    return {
        "skills_grouped":  dict(skills_grouped),
        "skills_flat":     skills_flat,
        "skills_required": skills_required,
        "skills_preferred": skills_preferred,
        "candidate_skills": candidate_skills,
        "match_method":    match_method,
    }


# ── Integration helper ─────────────────────────────────────────────────────────

def enrich_job(parsed_job: dict, **kwargs) -> dict:
    """
    Replace the basic skills in a parse_job() output with the advanced version.

    Args:
        parsed_job: dict returned by scraper/parser.py::parse_job()
        **kwargs:   forwarded to extract_skills_advanced (fuzzy, fuzzy_threshold, ...)

    Returns:
        Same dict with skills_grouped / skills_flat replaced and new keys added.
    """
    text = f"{parsed_job.get('description', '')} {parsed_job.get('requirements', '')}"
    advanced = extract_skills_advanced(text, **kwargs)
    return {
        **parsed_job,
        "skills_grouped":   advanced["skills_grouped"],
        "skills_flat":      advanced["skills_flat"],
        "skills_required":  advanced["skills_required"],
        "skills_preferred": advanced["skills_preferred"],
        "candidate_skills": advanced["candidate_skills"],
        "skill_match_method": advanced["match_method"],
    }


def enrich_all(parsed_jobs: list[dict], **kwargs) -> list[dict]:
    """Batch version of enrich_job()."""
    return [enrich_job(j, **kwargs) for j in parsed_jobs]


def compare_with_basic(text: str) -> dict:
    """
    Run both basic (parser.py) and advanced extraction and diff the results.
    Useful for evaluating alias / fuzzy coverage.
    """
    from scraper.parser import extract_skills, flat_skills
    basic    = set(flat_skills(extract_skills(text)))
    advanced = set(extract_skills_advanced(text)["skills_flat"])
    result   = extract_skills_advanced(text)
    return {
        "basic_only":    sorted(basic - advanced),
        "advanced_only": sorted(advanced - basic),
        "common":        sorted(basic & advanced),
        "fuzzy_matched": {k: v for k, v in result["match_method"].items() if v == "fuzzy"},
        "aliases_used":  {k: v for k, v in result["match_method"].items() if v == "alias"},
    }


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    sample = """
    We are looking for a Senior ML Engineer with strong Python and PyTorch skills.
    Experience with AWS SageMaker, Docker, and k8s is required.
    You must have solid knowledge of sklearn and wandb.
    Familiarity with LangChain and RAG pipelines is a plus.
    Experience with LangGraph or CrewAI would be a bonus.
    Fine tuning of LLMs and prompt eng experience preferred.
    Strong communication skills and team player mindset required.
    """

    print("-- Advanced extraction -------------------------------------------")
    result = extract_skills_advanced(sample)
    for cat, skills in result["skills_grouped"].items():
        print(f"  {cat}:")
        for s in skills:
            print(f"    {s['name']:30s}  count={s['count']}  [{s['requirement']}]")

    print(f"\n  Required:   {result['skills_required']}")
    print(f"  Preferred:  {result['skills_preferred']}")
    print(f"  Candidates: {result['candidate_skills']}")
    print(f"  Fuzzy hits: {[k for k,v in result['match_method'].items() if v=='fuzzy']}")
    print(f"  Alias hits: {[k for k,v in result['match_method'].items() if v=='alias']}")

    print("\n-- Diff vs basic parser ------------------------------------------")
    diff = compare_with_basic(sample)
    print(json.dumps(diff, indent=2))
