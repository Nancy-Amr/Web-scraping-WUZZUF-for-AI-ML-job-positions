"""
WUZZUF AI/ML Jobs Scraper
Uses Playwright for JS-rendered pages.

Install dependencies:
    pip install playwright pandas
    playwright install chromium
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import html as html_module
import time
import random
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from scraper.parser import parse_all
from db.models import JobsDB
from nlp.skill_extractor import enrich_all

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
        # Wait for any job detail link — /jobs/p/ is Wuzzuf's pattern for individual listings
        page.wait_for_selector("a[href*='/jobs/p/']", timeout=15_000)
        cards = page.query_selector_all("a[href*='/jobs/p/']")
        seen = set()
        links = []
        for card in cards:
            href = card.get_attribute("href") or ""
            if not href or href in seen:
                continue
            seen.add(href)
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
            page.wait_for_selector("h1", timeout=10_000)
        except PlaywrightTimeout:
            log.warning(f"Timeout loading job page: {url}")
            return None
        except Exception as e:
            log.warning(f"Error loading {url}: {e}")
            return None

        # ── JSON-LD structured data ───────────────────────────────────────────────
        ld_raw = page.evaluate("""() => {
            const el = document.querySelector('script[type="application/ld+json"]');
            try { return el ? JSON.parse(el.textContent) : {}; } catch(e) { return {}; }
        }""") or {}
        # JSON-LD may be a list; find the JobPosting entry
        if isinstance(ld_raw, list):
            ld = next((x for x in ld_raw if isinstance(x, dict) and "JobPosting" in str(x.get("@type", ""))), ld_raw[0] if ld_raw else {})
        else:
            ld = ld_raw if isinstance(ld_raw, dict) else {}

        def _strip_html(raw: str) -> str:
            text = html_module.unescape(raw)
            return re.sub(r"<[^>]+>", " ", text).strip()

        def _ld(key, default="N/A"):
            return ld.get(key) or default

        if not ld:
            log.warning(f"  No JSON-LD found on {url} — falling back to DOM for all fields")

        title = _ld("title") or self._safe_text(page, "h1")

        # company — JSON-LD first, then DOM link to /company/ page
        _hiring = _ld("hiringOrganization", {})
        company = (_hiring.get("name") if isinstance(_hiring, dict) else None) or \
                  page.evaluate("""() => {
                      for (const a of document.querySelectorAll('a[href*="/company/"]')) {
                          const t = a.textContent.trim();
                          if (t && t.length > 1) return t;
                      }
                      return null;
                  }""") or "N/A"

        # location — try addressRegion then addressLocality then DOM
        _addr = (_ld("jobLocation", {}) or {})
        if isinstance(_addr, list):
            _addr = _addr[0] if _addr else {}
        _addr = (_addr.get("address") or {}) if isinstance(_addr, dict) else {}
        location = (_addr.get("addressRegion") or _addr.get("addressLocality") or
                    _addr.get("addressCountry")) or \
                   page.evaluate("""() => {
                       for (const a of document.querySelectorAll('a[href*="/jobs/in/"]')) {
                           const t = a.textContent.trim();
                           if (t) return t;
                       }
                       return 'N/A';
                   }""") or "N/A"

        job_type    = _ld("employmentType")
        posted_date = _ld("datePosted")
        deadline    = _ld("validThrough")
        salary_obj  = _ld("baseSalary", {})
        salary      = str(salary_obj.get("value") or salary_obj.get("currency") or "N/A") \
                      if isinstance(salary_obj, dict) else "N/A"

        raw_desc    = _ld("description", "")
        desc_match  = re.search(r"Job Description</h1>(.*?)(?:<h1>|$)", raw_desc, re.DOTALL)
        req_match   = re.search(r"Job Requirements</h1>(.*?)(?:<h1>|$)", raw_desc, re.DOTALL)
        description  = _strip_html(desc_match.group(1)) if desc_match else _strip_html(raw_desc)
        requirements = _strip_html(req_match.group(1)) if req_match else "N/A"

        # ── Criteria (experience, career level) — text-based, survives CSS-in-JS renames ──
        criteria: dict[str, str] = page.evaluate("""() => {
            const result = {};
            const targets = new Set(['Years of Experience', 'Career Level', 'Job Type', 'Salary Range']);
            document.querySelectorAll('span, dt, li').forEach(el => {
                const label = el.textContent.trim().replace(/:$/, '');
                if (!targets.has(label)) return;
                const parent = el.parentElement;
                if (!parent) return;
                const next = el.nextElementSibling;
                let value = next ? next.textContent.trim() : '';
                if (!value) {
                    const children = Array.from(parent.children);
                    const idx = children.indexOf(el);
                    if (idx >= 0 && idx + 1 < children.length)
                        value = children[idx + 1].textContent.trim();
                }
                if (value && !result[label]) result[label] = value;
            });
            return result;
        }""") or {}

        experience   = criteria.get("Years of Experience", "N/A")
        career_level = criteria.get("Career Level", "N/A")

        # ── Categories & Skills — URL-pattern based, no CSS class dependency ──
        categories = page.evaluate("""() => {
            const cats = new Set();
            document.querySelectorAll('a[href*="category"]').forEach(a => {
                const t = a.textContent.trim();
                if (t && t.length < 60) cats.add(t);
            });
            return [...cats];
        }""") or []

        skills = page.evaluate("""() => {
            // Wuzzuf skill tags link to /skill/ pages; fall back to skills-section spans
            const byUrl = Array.from(document.querySelectorAll('a[href*="/skill/"]'))
                               .map(a => a.textContent.trim()).filter(t => t);
            if (byUrl.length) return byUrl;
            const headers = Array.from(document.querySelectorAll('h2,h3,h4,h5,strong'));
            const hdr = headers.find(h => /skills/i.test(h.textContent));
            if (hdr) {
                const container = hdr.closest('section') || hdr.parentElement;
                if (container)
                    return Array.from(container.querySelectorAll('a,span'))
                               .map(el => el.textContent.trim())
                               .filter(t => t.length > 1 && t.length < 40);
            }
            return [];
        }""") or []

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

                    # Check if results exist — text-based, no CSS class dependency
                    no_results = page.evaluate("""() => {
                        const body = document.body.innerText || '';
                        return /no (results|jobs) found/i.test(body) ||
                               /didn.t find any/i.test(body);
                    }""")
                    if no_results:
                        log.info("  No more results, stopping pagination.")
                        break

                    try:
                        job_links = self._get_job_links_from_page(page)
                    except Exception as e:
                        log.warning(f"  Failed to get links: {e}")
                        break

                    log.info(f"  Found {len(job_links)} jobs on this page.")
                    if not job_links:
                        log.warning("  0 links found — Wuzzuf page structure may have changed or bot-detection triggered.")
                        break

                    for link in job_links:
                        if link in seen_urls:
                            log.info(f"  [SKIP - duplicate] {link}")
                            continue
                        seen_urls.add(link)

                        log.info(f"  Scraping: {link}")
                        job = self._parse_job_page(page, link)

                        if job:
                            self.jobs.append(job)
                            log.info(f"  [OK] {job.title} @ {job.company}")

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
    # 1. Scrape and save raw JSON
    scraper = WuzzufScraper(headless=True)
    scraper.scrape()
    scraper.save()  # ← creates wuzzuf_ai_ml_jobs.json

    # 2. Load raw JSON and parse it
    if not scraper.jobs:
        log.warning("No jobs were scraped — skipping parse and DB insert.")
    else:
        with open("wuzzuf_ai_ml_jobs.json") as f:
            raw_jobs = json.load(f)

        parsed = parse_all(raw_jobs)
        parsed = enrich_all(parsed) 

        # 3. Save parsed version
        with open("wuzzuf_parsed_jobs.json", "w") as f:
            json.dump(parsed, f, indent=2)

        # 4. Insert into MongoDB
        with JobsDB() as db:
            result = db.insert_jobs(parsed)
            db.log_scrape_run(**result, queries=SEARCH_QUERIES)
            print(db.get_stats())