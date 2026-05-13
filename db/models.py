"""
db/models.py — MongoDB database layer for the WUZZUF job scraper.

Collections:
    jobs        — parsed job records (one document per job)
    scrape_logs — metadata about each scrape run

Usage:
    from db.models import JobsDB
    db = JobsDB()                          # connects to localhost by default
    db = JobsDB(uri="mongodb+srv://...")   # or Atlas connection string

    db.insert_jobs(parsed_jobs)
    db.get_all_jobs()
    db.search_by_skill("PyTorch")
    db.get_stats()
"""

import os
import logging
from datetime import datetime
from typing import Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, ConnectionFailure

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on env vars set externally

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME     = "wuzzuf_jobs"
COLLECTION_JOBS = "jobs"
COLLECTION_LOGS = "scrape_logs"


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class JobsDB:
    """
    All MongoDB operations for the scraper project.
    One instance per session — reuses the same connection.
    """

    def __init__(self, uri: str = DEFAULT_URI):
        """
        Connect to MongoDB and ensure indexes exist.

        Args:
            uri: MongoDB connection string.
                 Local  → "mongodb://localhost:27017/"
                 Atlas  → "mongodb+srv://user:pass@cluster.mongodb.net/"
        """
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Force connection check
            self.client.admin.command("ping")
            log.info("✓ Connected to MongoDB")
        except ConnectionFailure as e:
            log.error(f"✗ Could not connect to MongoDB: {e}")
            raise

        self.db         = self.client[DB_NAME]
        self.jobs       = self.db[COLLECTION_JOBS]
        self.logs       = self.db[COLLECTION_LOGS]

        self._ensure_indexes()

    # ── Indexes ───────────────────────────────────────────────────────────────

    def _ensure_indexes(self):
        """
        Create indexes once. Safe to call every startup (no-op if they exist).
        """
        # Unique index on job URL → prevents duplicate inserts
        self.jobs.create_index("job_url", unique=True, name="unique_job_url")

        # Fields we frequently filter/sort on
        self.jobs.create_index("scraped_at",          name="idx_scraped_at")
        self.jobs.create_index("skills_flat",          name="idx_skills")
        self.jobs.create_index("seniority",            name="idx_seniority")
        self.jobs.create_index("location.city",        name="idx_city")
        self.jobs.create_index("experience.min",       name="idx_exp_min")
        self.jobs.create_index("salary.min",           name="idx_salary_min")
        self.jobs.create_index([("title", ASCENDING),
                                ("company", ASCENDING)], name="idx_title_company")

        log.info("✓ Indexes ensured")

    # ══════════════════════════════════════════════════════════════════════════
    # INSERT
    # ══════════════════════════════════════════════════════════════════════════

    def insert_job(self, job: dict) -> bool:
        """
        Insert a single parsed job. Skips silently if URL already exists.

        Returns:
            True if inserted, False if duplicate.
        """
        try:
            self.jobs.insert_one(job)
            log.info(f"  ✓ Inserted: {job.get('title')} @ {job.get('company')}")
            return True
        except DuplicateKeyError:
            log.debug(f"  ~ Duplicate skipped: {job.get('job_url')}")
            return False
        except Exception as e:
            log.error(f"  ✗ Insert failed: {e}")
            return False

    def insert_jobs(self, jobs: list[dict]) -> dict:
        """
        Bulk insert a list of parsed jobs. Skips duplicates automatically.

        Returns:
            {"inserted": N, "duplicates": M, "errors": K}
        """
        inserted = duplicates = errors = 0

        for job in jobs:
            try:
                self.jobs.insert_one(job)
                inserted += 1
            except DuplicateKeyError:
                duplicates += 1
            except Exception as e:
                log.error(f"Insert error for {job.get('job_url')}: {e}")
                errors += 1

        summary = {"inserted": inserted, "duplicates": duplicates, "errors": errors}
        log.info(f"Bulk insert done → {summary}")
        return summary

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY
    # ══════════════════════════════════════════════════════════════════════════

    def get_all_jobs(self, limit: int = 0) -> list[dict]:
        """Return all jobs, newest first. limit=0 means no limit."""
        cursor = self.jobs.find({}, {"_id": 0}).sort("scraped_at", DESCENDING)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def search_by_skill(self, skill: str) -> list[dict]:
        """
        Find all jobs that mention a specific skill.

        Args:
            skill: e.g. "PyTorch", "AWS", "Docker"
        """
        results = self.jobs.find(
            {"skills_flat": {"$regex": skill, "$options": "i"}},
            {"_id": 0}
        ).sort("scraped_at", DESCENDING)
        return list(results)

    def search_by_skills(self, skills: list[str], match_all: bool = False) -> list[dict]:
        """
        Find jobs matching multiple skills.

        Args:
            skills:    list of skill names e.g. ["PyTorch", "Docker"]
            match_all: True  → job must have ALL skills ($and)
                       False → job must have ANY skill ($or)
        """
        conditions = [
            {"skills_flat": {"$regex": s, "$options": "i"}} for s in skills
        ]
        operator = "$and" if match_all else "$or"
        results = self.jobs.find({operator: conditions}, {"_id": 0})
        return list(results)

    def search_by_seniority(self, level: str) -> list[dict]:
        """
        Filter by seniority level.
        level: "Junior" | "Mid" | "Senior" | "Intern" | "Manager"
        """
        results = self.jobs.find(
            {"seniority": {"$regex": level, "$options": "i"}},
            {"_id": 0}
        )
        return list(results)

    def search_by_city(self, city: str) -> list[dict]:
        """Filter jobs by city name."""
        results = self.jobs.find(
            {"location.city": {"$regex": city, "$options": "i"}},
            {"_id": 0}
        )
        return list(results)

    def search_remote(self) -> list[dict]:
        """Return all remote or hybrid jobs."""
        results = self.jobs.find(
            {"$or": [{"location.remote": True}, {"location.hybrid": True}]},
            {"_id": 0}
        )
        return list(results)

    def search_by_experience(self, min_years: int, max_years: int) -> list[dict]:
        """
        Find jobs within an experience range.

        Example: min_years=2, max_years=5 → jobs requiring 2-5 years.
        """
        results = self.jobs.find(
            {
                "experience.min": {"$gte": min_years},
                "experience.max": {"$lte": max_years},
            },
            {"_id": 0}
        )
        return list(results)

    def search_by_salary(self, min_salary: int, currency: str = "EGP") -> list[dict]:
        """Find jobs with salary >= min_salary in the given currency."""
        results = self.jobs.find(
            {
                "salary.min":      {"$gte": min_salary},
                "salary.currency": currency,
            },
            {"_id": 0}
        )
        return list(results)

    def full_text_search(self, keyword: str) -> list[dict]:
        """
        Search across title, description, and requirements fields.
        Uses regex — good enough without a full-text index.
        """
        pattern = {"$regex": keyword, "$options": "i"}
        results = self.jobs.find(
            {"$or": [
                {"title":        pattern},
                {"description":  pattern},
                {"requirements": pattern},
                {"company":      pattern},
            ]},
            {"_id": 0}
        )
        return list(results)

    def get_recent_jobs(self, days: int = 7) -> list[dict]:
        """Return jobs scraped within the last N days."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        results = self.jobs.find(
            {"scraped_at": {"$gte": cutoff}},
            {"_id": 0}
        ).sort("scraped_at", DESCENDING)
        return list(results)

    def get_job_by_url(self, url: str) -> Optional[dict]:
        """Fetch a single job by its WUZZUF URL."""
        return self.jobs.find_one({"job_url": url}, {"_id": 0})

    # ══════════════════════════════════════════════════════════════════════════
    # ANALYTICS QUERIES
    # ══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """
        Return high-level stats about the jobs collection.
        Used by the dashboard.
        """
        total = self.jobs.count_documents({})

        # Top skills (unwind skills_flat array, count each)
        top_skills = list(self.jobs.aggregate([
            {"$unwind": "$skills_flat"},
            {"$group": {"_id": "$skills_flat", "count": {"$sum": 1}}},
            {"$sort": {"count": DESCENDING}},
            {"$limit": 20},
        ]))

        # Jobs by seniority
        by_seniority = list(self.jobs.aggregate([
            {"$group": {"_id": "$seniority", "count": {"$sum": 1}}},
            {"$sort": {"count": DESCENDING}},
        ]))

        # Jobs by city
        by_city = list(self.jobs.aggregate([
            {"$group": {"_id": "$location.city", "count": {"$sum": 1}}},
            {"$sort": {"count": DESCENDING}},
            {"$limit": 10},
        ]))

        # Jobs by company
        by_company = list(self.jobs.aggregate([
            {"$group": {"_id": "$company", "count": {"$sum": 1}}},
            {"$sort": {"count": DESCENDING}},
            {"$limit": 10},
        ]))

        # Remote vs on-site
        remote_count  = self.jobs.count_documents({"location.remote": True})
        hybrid_count  = self.jobs.count_documents({"location.hybrid": True})
        onsite_count  = total - remote_count - hybrid_count

        # Average salary (where available)
        salary_agg = list(self.jobs.aggregate([
            {"$match": {"salary.min": {"$exists": True, "$ne": None}}},
            {"$group": {
                "_id": None,
                "avg_min": {"$avg": "$salary.min"},
                "avg_max": {"$avg": "$salary.max"},
            }},
        ]))
        avg_salary = salary_agg[0] if salary_agg else {}

        return {
            "total_jobs":   total,
            "top_skills":   [{"skill": s["_id"], "count": s["count"]} for s in top_skills],
            "by_seniority": [{"level": s["_id"], "count": s["count"]} for s in by_seniority],
            "by_city":      [{"city":  s["_id"], "count": s["count"]} for s in by_city],
            "by_company":   [{"company": s["_id"], "count": s["count"]} for s in by_company],
            "work_type": {
                "remote":  remote_count,
                "hybrid":  hybrid_count,
                "on_site": onsite_count,
            },
            "avg_salary_egp": {
                "min": round(avg_salary.get("avg_min", 0)),
                "max": round(avg_salary.get("avg_max", 0)),
            },
        }

    def skills_trend(self, top_n: int = 15) -> list[dict]:
        """
        Return the top N most in-demand skills across all jobs.
        Perfect for a bar chart in the dashboard.
        """
        pipeline = [
            {"$unwind": "$skills_flat"},
            {"$group": {"_id": "$skills_flat", "count": {"$sum": 1}}},
            {"$sort": {"count": DESCENDING}},
            {"$limit": top_n},
        ]
        results = list(self.jobs.aggregate(pipeline))
        return [{"skill": r["_id"], "count": r["count"]} for r in results]

    def skills_by_seniority(self) -> dict:
        """
        For each seniority level, return the top 10 required skills.
        Useful to show 'Junior jobs need X, Senior jobs need Y'.
        """
        levels = ["Junior", "Mid", "Senior", "Intern", "Manager"]
        result = {}
        for level in levels:
            pipeline = [
                {"$match": {"seniority": level}},
                {"$unwind": "$skills_flat"},
                {"$group": {"_id": "$skills_flat", "count": {"$sum": 1}}},
                {"$sort": {"count": DESCENDING}},
                {"$limit": 10},
            ]
            skills = list(self.jobs.aggregate(pipeline))
            result[level] = [{"skill": s["_id"], "count": s["count"]} for s in skills]
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # SCRAPE LOG
    # ══════════════════════════════════════════════════════════════════════════

    def log_scrape_run(self, inserted: int, duplicates: int, errors: int, queries: list[str]):
        """Record metadata about a scrape run for tracking purposes."""
        self.logs.insert_one({
            "timestamp":  datetime.now().isoformat(),
            "inserted":   inserted,
            "duplicates": duplicates,
            "errors":     errors,
            "queries":    queries,
        })

    def get_scrape_history(self) -> list[dict]:
        """Return all past scrape run logs, newest first."""
        return list(self.logs.find({}, {"_id": 0}).sort("timestamp", DESCENDING))

    # ══════════════════════════════════════════════════════════════════════════
    # MAINTENANCE
    # ══════════════════════════════════════════════════════════════════════════

    def delete_job(self, url: str) -> bool:
        """Delete a job by its URL."""
        result = self.jobs.delete_one({"job_url": url})
        return result.deleted_count > 0

    def clear_all_jobs(self, confirm: bool = False):
        """Drop all jobs. Requires confirm=True as a safety guard."""
        if not confirm:
            log.warning("Pass confirm=True to clear all jobs.")
            return
        self.jobs.drop()
        self._ensure_indexes()
        log.info("All jobs cleared.")

    def count(self) -> int:
        """Return total number of jobs in the collection."""
        return self.jobs.count_documents({})

    def close(self):
        """Close the MongoDB connection."""
        self.client.close()
        log.info("MongoDB connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# Quick test — run: python db/models.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import json

    sample_jobs = [
        {
            "title": "Senior ML Engineer",
            "company": "Acme AI",
            "job_url": "https://wuzzuf.net/jobs/p/test-001",
            "scraped_at": datetime.now().isoformat(),
            "location": {"city": "Cairo", "country": "Egypt", "remote": False, "hybrid": False},
            "experience": {"min": 3, "max": 5, "raw": "3 - 5 Years"},
            "salary": {"min": 25000, "max": 35000, "currency": "EGP", "period": "Month"},
            "seniority": "Senior",
            "skills_flat": ["Python", "PyTorch", "AWS", "Docker", "MLflow"],
            "skills_grouped": {
                "Languages": ["Python"],
                "ML Frameworks": ["PyTorch"],
                "MLOps & Cloud": ["AWS", "Docker", "MLflow"],
            },
            "description": "Build ML pipelines at scale.",
            "requirements": "3-5 years experience with Python and PyTorch.",
            "job_type": "Full Time",
            "career_level": "Senior",
            "posted_date": "2025-05-10",
            "deadline": "2025-06-01",
            "categories": ["Technology"],
        },
        {
            "title": "Junior Data Scientist",
            "company": "DataCorp",
            "job_url": "https://wuzzuf.net/jobs/p/test-002",
            "scraped_at": datetime.now().isoformat(),
            "location": {"city": "Cairo", "country": "Egypt", "remote": True, "hybrid": False},
            "experience": {"min": 0, "max": 2, "raw": "0 - 2 Years"},
            "salary": {"min": 8000, "max": 12000, "currency": "EGP", "period": "Month"},
            "seniority": "Junior",
            "skills_flat": ["Python", "Scikit-learn", "Pandas", "SQL"],
            "skills_grouped": {
                "Languages": ["Python", "SQL"],
                "ML Frameworks": ["Scikit-learn"],
                "Data Engineering": ["Pandas"],
            },
            "description": "Analyze data and build models for our products.",
            "requirements": "Fresh grad or 0-2 years experience.",
            "job_type": "Full Time",
            "career_level": "Junior",
            "posted_date": "2025-05-12",
            "deadline": "2025-06-15",
            "categories": ["Data Science"],
        },
    ]

    with JobsDB() as db:
        # Insert
        result = db.insert_jobs(sample_jobs)
        print(f"\nInsert result: {result}")

        # Count
        print(f"Total jobs: {db.count()}")

        # Search
        pytorch_jobs = db.search_by_skill("PyTorch")
        print(f"\nJobs requiring PyTorch: {len(pytorch_jobs)}")

        remote_jobs = db.search_remote()
        print(f"Remote jobs: {len(remote_jobs)}")

        # Stats
        stats = db.get_stats()
        print(f"\n── Stats ──────────────────────────────")
        print(f"Total:       {stats['total_jobs']}")
        print(f"Top skills:  {stats['top_skills'][:5]}")
        print(f"By city:     {stats['by_city']}")
        print(f"Work type:   {stats['work_type']}")