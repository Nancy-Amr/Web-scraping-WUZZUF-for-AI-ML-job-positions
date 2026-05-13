"""
parser.py — Clean and structure raw job data from the WUZZUF scraper.

Responsibilities:
    1. Extract skills mentioned in description/requirements text
    2. Parse and normalize experience ranges
    3. Normalize location strings
    4. Parse relative/absolute dates into ISO format
    5. Parse salary ranges into structured fields
    6. Detect seniority level from title + career level
    7. parse_job() — run all of the above on a raw Job dict at once
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. SKILLS EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

# Master skill dictionary grouped by category.
# Add / remove skills here freely — parser will auto-detect them in text.
SKILL_TAXONOMY: dict[str, list[str]] = {
    "Languages": [
        "Python", "R", "SQL", "Java", "C++", "C#", "Scala", "Julia",
        "JavaScript", "TypeScript", "Go", "Rust", "MATLAB", "Bash",
    ],
    "ML Frameworks": [
        "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "XGBoost",
        "LightGBM", "CatBoost", "Hugging Face", "Transformers", "JAX",
        "MXNet", "FastAI", "ONNX",
    ],
    "Generative AI": [
        "LangChain", "LlamaIndex", "OpenAI", "GPT", "BERT", "LLM",
        "Diffusion Models", "GANs", "VAE", "Stable Diffusion",
        "Retrieval Augmented Generation", "RAG", "Fine-tuning",
        "Prompt Engineering",
    ],
    "Computer Vision": [
        "OpenCV", "YOLO", "ResNet", "VGG", "EfficientNet", "CLIP",
        "SAM", "Detectron2", "Albumentations", "PIL", "Pillow",
    ],
    "NLP": [
        "NLTK", "spaCy", "Gensim", "FastText", "Word2Vec", "BERT",
        "Sentence Transformers", "TextBlob", "Tokenization", "NER",
    ],
    "MLOps & Cloud": [
        "MLflow", "DVC", "Weights & Biases", "W&B", "Kubeflow",
        "Airflow", "Prefect", "ZenML", "BentoML", "Seldon",
        "AWS", "GCP", "Azure", "SageMaker", "Vertex AI",
        "Docker", "Kubernetes", "Terraform", "CI/CD",
    ],
    "Data Engineering": [
        "Spark", "Kafka", "Flink", "Hadoop", "Hive", "Airflow",
        "dbt", "Snowflake", "BigQuery", "Redshift", "Databricks",
        "Pandas", "Polars", "NumPy", "Dask",
    ],
    "Databases": [
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
        "Cassandra", "SQLite", "Oracle", "SQL Server", "Pinecone",
        "Weaviate", "Chroma", "FAISS",
    ],
    "Visualization": [
        "Matplotlib", "Seaborn", "Plotly", "Tableau", "Power BI",
        "Streamlit", "Gradio", "Dash",
    ],
    "Soft Skills": [
        "Communication", "Teamwork", "Leadership", "Problem Solving",
        "Research", "Analytical", "Attention to Detail",
    ],
}

# Flat lookup: lowercase skill name → (canonical name, category)
_SKILL_LOOKUP: dict[str, tuple[str, str]] = {}
for _cat, _skills in SKILL_TAXONOMY.items():
    for _skill in _skills:
        _SKILL_LOOKUP[_skill.lower()] = (_skill, _cat)


def extract_skills(text: str) -> dict[str, list[str]]:
    """
    Scan text and return found skills grouped by category.

    Args:
        text: Raw job description or requirements string.

    Returns:
        {
          "ML Frameworks": ["PyTorch", "TensorFlow"],
          "Cloud": ["AWS"],
          ...
        }
    """
    if not text:
        return {}

    text_lower = text.lower()
    found: dict[str, list[str]] = {}

    for skill_lower, (canonical, category) in _SKILL_LOOKUP.items():
        # Use word boundaries so "R" doesn't match inside "research"
        pattern = r"\b" + re.escape(skill_lower) + r"\b"
        if re.search(pattern, text_lower):
            found.setdefault(category, [])
            if canonical not in found[category]:
                found[category].append(canonical)

    return found


def flat_skills(skills_dict: dict[str, list[str]]) -> list[str]:
    """Flatten the grouped skills dict into a simple list."""
    return [skill for skills in skills_dict.values() for skill in skills]


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXPERIENCE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_experience(raw: str) -> dict:
    """
    Parse experience strings into structured min/max years.

    Examples:
        "3 - 5 Years"  → {"min": 3, "max": 5,    "raw": "3 - 5 Years"}
        "1+ Year"      → {"min": 1, "max": None,  "raw": "1+ Year"}
        "Fresh Grad"   → {"min": 0, "max": 0,     "raw": "Fresh Grad"}
        "N/A"          → {"min": None, "max": None,"raw": "N/A"}
    """
    result = {"min": None, "max": None, "raw": raw}

    if not raw or raw.strip() in ("N/A", "", "—"):
        return result

    text = raw.lower().strip()

    # Fresh graduate
    if any(k in text for k in ("fresh", "no experience", "0 year")):
        result.update({"min": 0, "max": 0})
        return result

    # Range: "3 - 5" or "3–5" or "3 to 5"
    range_match = re.search(r"(\d+)\s*[-–to]+\s*(\d+)", text)
    if range_match:
        result.update({"min": int(range_match.group(1)), "max": int(range_match.group(2))})
        return result

    # "1+" or "5 +"
    plus_match = re.search(r"(\d+)\s*\+", text)
    if plus_match:
        result.update({"min": int(plus_match.group(1)), "max": None})
        return result

    # Single number: "5 years"
    single_match = re.search(r"(\d+)", text)
    if single_match:
        val = int(single_match.group(1))
        result.update({"min": val, "max": val})

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOCATION NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════

REMOTE_KEYWORDS = {"remote", "work from home", "wfh", "hybrid", "online"}
EGYPT_CITIES = {
    "cairo", "giza", "alexandria", "mansoura", "tanta", "zagazig",
    "ismailia", "suez", "luxor", "aswan", "hurghada", "nasr city",
    "heliopolis", "maadi", "new cairo", "6th of october", "10th of ramadan",
    "obour", "sheikh zayed", "new capital",
}


def parse_location(raw: str) -> dict:
    """
    Normalize a raw location string.

    Returns:
        {
          "city": "Cairo",
          "country": "Egypt",
          "remote": False,
          "hybrid": False,
          "raw": "Cairo, Egypt"
        }
    """
    result = {
        "city": None,
        "country": None,
        "remote": False,
        "hybrid": False,
        "raw": raw,
    }

    if not raw or raw.strip() == "N/A":
        return result

    lower = raw.lower()

    result["remote"] = any(k in lower for k in ("remote", "work from home", "wfh"))
    result["hybrid"] = "hybrid" in lower

    # Try to extract city
    for city in EGYPT_CITIES:
        if city in lower:
            result["city"] = city.title()
            result["country"] = "Egypt"
            break

    # Fallback: split on comma
    if not result["city"]:
        parts = [p.strip() for p in raw.split(",")]
        if parts:
            result["city"] = parts[0]
        if len(parts) > 1:
            result["country"] = parts[-1]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_date(raw: str, reference: Optional[datetime] = None) -> Optional[str]:
    """
    Convert relative or absolute date strings to ISO 8601 format.

    Examples:
        "3 days ago"   → "2025-05-10"
        "2 weeks ago"  → "2025-04-29"
        "1 month ago"  → "2025-04-13"
        "2025-04-01"   → "2025-04-01"
        "N/A"          → None
    """
    if not raw or raw.strip() in ("N/A", "—", ""):
        return None

    ref = reference or datetime.now()
    text = raw.lower().strip()

    # Already an ISO date
    iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if iso_match:
        return iso_match.group(1)

    # Relative: "X days/weeks/months ago"
    num_match = re.search(r"(\d+)\s*(day|week|month|hour|minute)", text)
    if num_match:
        n = int(num_match.group(1))
        unit = num_match.group(2)
        if unit == "day":
            delta = timedelta(days=n)
        elif unit == "week":
            delta = timedelta(weeks=n)
        elif unit == "month":
            delta = timedelta(days=n * 30)
        elif unit == "hour":
            delta = timedelta(hours=n)
        else:
            delta = timedelta(minutes=n)
        return (ref - delta).strftime("%Y-%m-%d")

    # "yesterday"
    if "yesterday" in text:
        return (ref - timedelta(days=1)).strftime("%Y-%m-%d")

    # "today"
    if "today" in text:
        return ref.strftime("%Y-%m-%d")

    # Try common formats
    for fmt in ("%d %b %Y", "%B %d, %Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    log.debug(f"Could not parse date: '{raw}'")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 5. SALARY PARSER
# ══════════════════════════════════════════════════════════════════════════════

CURRENCIES = {"EGP", "USD", "EUR", "GBP", "SAR", "AED"}


def parse_salary(raw: str) -> Optional[dict]:
    """
    Parse salary strings into structured data.

    Examples:
        "EGP 15,000 - 20,000 / Month" → {"min": 15000, "max": 20000,
                                           "currency": "EGP", "period": "Month"}
        "Confidential"                 → None
        "N/A"                          → None
    """
    if not raw or raw.strip().lower() in ("n/a", "confidential", "—", ""):
        return None

    result = {"min": None, "max": None, "currency": "EGP", "period": "Month", "raw": raw}

    # Currency
    for cur in CURRENCIES:
        if cur in raw.upper():
            result["currency"] = cur
            break

    # Period
    lower = raw.lower()
    if "year" in lower or "annual" in lower:
        result["period"] = "Year"
    elif "hour" in lower:
        result["period"] = "Hour"
    else:
        result["period"] = "Month"

    # Extract numbers (handles commas like "15,000")
    numbers = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", raw) if n.replace(",", "").isdigit()]

    if len(numbers) >= 2:
        result["min"] = min(numbers[:2])
        result["max"] = max(numbers[:2])
    elif len(numbers) == 1:
        result["min"] = numbers[0]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 6. SENIORITY DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

SENIORITY_KEYWORDS: dict[str, list[str]] = {
    "Intern":    ["intern", "internship", "trainee"],
    "Junior":    ["junior", "jr.", "jr ", "entry", "associate", "fresh grad", "graduate"],
    "Mid":       ["mid", "mid-level", "intermediate"],
    "Senior":    ["senior", "sr.", "sr ", "lead", "principal", "staff"],
    "Manager":   ["manager", "head of", "director", "vp ", "vice president"],
}


def detect_seniority(title: str, career_level: str, experience: dict) -> str:
    """
    Infer seniority from job title, career level field, and experience years.

    Returns one of: "Intern" | "Junior" | "Mid" | "Senior" | "Manager" | "Unknown"
    """
    combined = f"{title} {career_level}".lower()

    for level, keywords in SENIORITY_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return level

    # Fallback on experience years
    min_exp = experience.get("min")
    if min_exp is not None:
        if min_exp == 0:
            return "Junior"
        elif min_exp <= 2:
            return "Junior"
        elif min_exp <= 5:
            return "Mid"
        else:
            return "Senior"

    return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# 7. MASTER PARSER — runs all of the above on one raw job dict
# ══════════════════════════════════════════════════════════════════════════════

def parse_job(raw: dict) -> dict:
    """
    Take a raw Job dict (as returned by the scraper) and return a fully
    cleaned, structured record ready for storage or analysis.

    Args:
        raw: dict with keys matching the Job dataclass fields.

    Returns:
        Enriched dict with all parsed/normalized fields added.
    """
    full_text = f"{raw.get('description', '')} {raw.get('requirements', '')}"

    # Run all parsers
    experience  = parse_experience(raw.get("experience_years", ""))
    location    = parse_location(raw.get("location", ""))
    salary      = parse_salary(raw.get("salary", ""))
    posted      = parse_date(raw.get("posted_date", ""))
    deadline    = parse_date(raw.get("deadline", ""))
    skills_dict = extract_skills(full_text)

    # Also merge any skill tags the scraper already found
    for tag in raw.get("skills", []):
        tag_lower = tag.lower()
        if tag_lower in _SKILL_LOOKUP:
            canonical, category = _SKILL_LOOKUP[tag_lower]
            skills_dict.setdefault(category, [])
            if canonical not in skills_dict[category]:
                skills_dict[category].append(canonical)

    seniority = detect_seniority(
        raw.get("title", ""),
        raw.get("career_level", ""),
        experience,
    )

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "title":           raw.get("title"),
        "company":         raw.get("company"),
        "job_url":         raw.get("job_url"),
        "scraped_at":      raw.get("scraped_at"),

        # ── Parsed fields ─────────────────────────────────────────────────────
        "location":        location,
        "experience":      experience,
        "salary":          salary,
        "posted_date":     posted,
        "deadline":        deadline,
        "seniority":       seniority,
        "job_type":        raw.get("job_type"),
        "career_level":    raw.get("career_level"),

        # ── Skills (grouped + flat) ───────────────────────────────────────────
        "skills_grouped":  skills_dict,
        "skills_flat":     flat_skills(skills_dict),

        # ── Raw text (kept for NLP next module) ───────────────────────────────
        "description":     raw.get("description"),
        "requirements":    raw.get("requirements"),
        "categories":      raw.get("categories", []),
    }


def parse_all(raw_jobs: list[dict]) -> list[dict]:
    """Parse a list of raw job dicts. Skips and logs any that fail."""
    parsed = []
    for i, job in enumerate(raw_jobs):
        try:
            parsed.append(parse_job(job))
        except Exception as e:
            log.warning(f"Failed to parse job {i} ({job.get('title', '?')}): {e}")
    log.info(f"Parsed {len(parsed)}/{len(raw_jobs)} jobs successfully.")
    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# Quick test — run: python parser.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sample = {
        "title": "Senior Machine Learning Engineer",
        "company": "Acme AI",
        "location": "Cairo, Egypt",
        "experience_years": "3 - 5 Years",
        "career_level": "Senior",
        "salary": "EGP 25,000 - 35,000 / Month",
        "posted_date": "3 days ago",
        "deadline": "2025-06-01",
        "job_url": "https://wuzzuf.net/jobs/p/example",
        "description": (
            "We are looking for a Senior ML Engineer with strong Python and PyTorch skills. "
            "Experience with AWS SageMaker, Docker, and Kubernetes is required. "
            "Familiarity with LangChain and RAG pipelines is a plus. "
            "You will work on large-scale NLP and computer vision systems."
        ),
        "requirements": (
            "3-5 years of experience in machine learning. "
            "Proficient in Scikit-learn, XGBoost, and Pandas. "
            "Experience with MLflow for experiment tracking. "
            "Strong SQL and data engineering background preferred."
        ),
        "skills": ["Python", "TensorFlow", "OpenCV"],
        "categories": ["Technology", "Computer Software"],
        "scraped_at": datetime.now().isoformat(),
    }

    result = parse_job(sample)
    print(json.dumps(result, indent=2, default=str))