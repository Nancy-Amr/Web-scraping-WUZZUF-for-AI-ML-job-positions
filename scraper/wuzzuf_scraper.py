"""
WUZZUF AI/ML Jobs Scraper
Uses Playwright for JS-rendered pages.

Install dependencies:
    pip install playwright pandas
    playwright install chromium
"""

import json
import time
import random
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from parser import parse_all
import json

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "https://wuzzuf.net/search/jobs/"
SEARCH_QUERIES = ["machine learning engineer", "AI engineer", "data scientist", "deep learning", "MLOps"]
MAX_PAGES_PER_QUERY = 5          # pages to scrape per search term
DELAY_BETWEEN_PAGES = (2, 4)     # random sleep range in seconds (be polite!)
OUTPUT_CSV = "wuzzuf_ai_ml_jobs.csv"
OUTPUT_JSON = "wuzzuf_ai_ml_jobs.json"


# ── Data Model ────────────────────────────────────────────────────────────────
@dataclass
class Job:
    title: str
    company: str
    location: str
    job_type: str                  # Full Time / Part Time / etc.
    experience_years: str
    career_level: str
    salary: str
    posted_date: str
    deadline: str
    job_url: str
    description: str
    requirements: str
    skills: list[str]
    categories: list[str]
    scraped_at: str = ""

    def __post_init__(self):
        self.scraped_at = datetime.now().isoformat()


# ── Scraper ───────────────────────────────────────────────────────────────────
class WuzzufScraper:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.jobs: list[Job] = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _random_delay(self):
        """Polite random delay between requests."""
        t = random.uniform(*DELAY_BETWEEN_PAGES)
        log.info(f"Sleeping {t:.1f}s ...")
        time.sleep(t)

    def _safe_text(self, page, selector: str, default: str = "N/A") -> str:
        """Extract text safely; return default if element missing."""
        try:
            el = page.query_selector(selector)
            return el.inner_text().strip() if el else default
        except Exception:
            return default

    def _safe_list(self, page, selector: str) -> list[str]:
        """Extract list of texts from multiple matching elements."""
        try:
            els = page.query_selector_all(selector)
            return [el.inner_text().strip() for el in els if el.inner_text().strip()]
        except Exception:
            return []

    # ── Search Results Page ───────────────────────────────────────────────────

    def _get_job_links_from_page(self, page) -> list[str]:
        """Return all job detail URLs from a search results page."""
        page.wait_for_selector("div.css-pkv5jc", timeout=15_000)   # job card container
        cards = page.query_selector_all("h2.css-m604qf a")         # job title links
        links = []
        for card in cards:
            href = card.get_attribute("href")
            if href:
                links.append("https://wuzzuf.net" + href if href.startswith("/") else href)
        return links

    def _build_search_url(self, query: str, page_num: int) -> str:
        """Build paginated search URL for a query."""
        encoded = query.replace(" ", "%20")
        return f"{BASE_URL}?q={encoded}&a=hpb&start={page_num * 15}"

    # ── Job Detail Page ───────────────────────────────────────────────────────

    def _parse_job_page(self, page, url: str) -> Optional[Job]:
        """Visit a job detail page and extract all fields."""
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            page.wait_for_selector("h1.css-f4ls5e", timeout=10_000)
        except PlaywrightTimeout:
            log.warning(f"Timeout loading job page: {url}")
            return None
        except Exception as e:
            log.warning(f"Error loading {url}: {e}")
            return None

        # ── Basic fields ──────────────────────────────────────────────────────
        title    = self._safe_text(page, "h1.css-f4ls5e")
        company  = self._safe_text(page, "a.css-17s97q8")
        location = self._safe_text(page, "span.css-5wys0k")

        # ── Job criteria items (type, experience, level, salary) ──────────────
        criteria_labels = self._safe_list(page, "div.css-1lh32fc span.css-4xky9y")
        criteria_values = self._safe_list(page, "div.css-1lh32fc span.css-o171kl, div.css-1lh32fc a.css-o171kl")

        criteria = dict(zip(criteria_labels, criteria_values))

        job_type        = criteria.get("Job Type",       "N/A")
        experience      = criteria.get("Years of Experience", "N/A")
        career_level    = criteria.get("Career Level",   "N/A")
        salary          = criteria.get("Salary",         "N/A")

        # ── Dates ─────────────────────────────────────────────────────────────
        posted_date = self._safe_text(page, "time")
        deadline    = self._safe_text(page, "span.css-4c4ojb")

        # ── Description & Requirements ────────────────────────────────────────
        description  = self._safe_text(page, "div.css-1uobp1k")
        requirements = self._safe_text(page, "div.css-dtmqe8")

        # ── Skills tags ───────────────────────────────────────────────────────
        skills     = self._safe_list(page, "a.css-b1l870")
        categories = self._safe_list(page, "a.css-o171kl[href*='category']")

        return Job(
            title=title,
            company=company,
            location=location,
            job_type=job_type,
            experience_years=experience,
            career_level=career_level,
            salary=salary,
            posted_date=posted_date,
            deadline=deadline,
            job_url=url,
            description=description,
            requirements=requirements,
            skills=skills,
            categories=categories,
        )

    # ── Main Scrape Loop ──────────────────────────────────────────────────────

    def scrape(self):
        seen_urls: set[str] = set()     # deduplicate across queries

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()

            for query in SEARCH_QUERIES:
                log.info(f"\n{'='*50}")
                log.info(f"Query: '{query}'")
                log.info(f"{'='*50}")

                for page_num in range(MAX_PAGES_PER_QUERY):
                    search_url = self._build_search_url(query, page_num)
                    log.info(f"  Page {page_num + 1}: {search_url}")

                    try:
                        page.goto(search_url, wait_until="domcontentloaded", timeout=20_000)
                    except PlaywrightTimeout:
                        log.warning("  Search page timed out, skipping.")
                        break

                    # Check if results exist
                    no_results = page.query_selector("div.css-1hfxbql")
                    if no_results:
                        log.info("  No more results, stopping pagination.")
                        break

                    try:
                        job_links = self._get_job_links_from_page(page)
                    except Exception as e:
                        log.warning(f"  Failed to get links: {e}")
                        break

                    log.info(f"  Found {len(job_links)} jobs on this page.")

                    for link in job_links:
                        if link in seen_urls:
                            log.info(f"  [SKIP - duplicate] {link}")
                            continue
                        seen_urls.add(link)

                        log.info(f"  Scraping: {link}")
                        job = self._parse_job_page(page, link)

                        if job:
                            self.jobs.append(job)
                            log.info(f"  ✓ {job.title} @ {job.company}")

                        self._random_delay()

                    self._random_delay()

            browser.close()

        log.info(f"\nTotal jobs scraped: {len(self.jobs)}")

    # ── Export ────────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for job in self.jobs:
            d = asdict(job)
            # Flatten lists to comma-separated strings for CSV
            d["skills"]     = ", ".join(d["skills"])
            d["categories"] = ", ".join(d["categories"])
            records.append(d)
        return pd.DataFrame(records)

    def save(self):
        if not self.jobs:
            log.warning("No jobs to save.")
            return

        # CSV
        df = self.to_dataframe()
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        log.info(f"Saved CSV → {OUTPUT_CSV}")

        # JSON (preserves lists)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump([asdict(j) for j in self.jobs], f, ensure_ascii=False, indent=2)
        log.info(f"Saved JSON → {OUTPUT_JSON}")

        # Quick summary
        df_raw = pd.DataFrame([asdict(j) for j in self.jobs])
        print("\n── Top companies hiring ──────────────────────────")
        print(df_raw["company"].value_counts().head(10).to_string())
        print("\n── Jobs by location ──────────────────────────────")
        print(df_raw["location"].value_counts().head(10).to_string())
        print("\n── Jobs by career level ──────────────────────────")
        print(df_raw["career_level"].value_counts().to_string())


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scraper = WuzzufScraper(headless=True)
    scraper.scrape()
    scraper.save()
    with open("wuzzuf_ai_ml_jobs.json") as f:
        raw_jobs = json.load(f)

    parsed = parse_all(raw_jobs)

    with open("wuzzuf_parsed_jobs.json", "w") as f:
        json.dump(parsed, f, indent=2)