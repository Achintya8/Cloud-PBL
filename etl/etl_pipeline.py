"""
etl/etl_pipeline.py
===================
Production ETL Pipeline for Areca Nut Price Prediction System.

Responsibilities:
  1. Scrape daily wholesale prices from Agmarknet (agmarknet.gov.in)
  2. Scrape regional APMC portal data as secondary source
  3. Fetch weather data (rainfall, humidity, temperature) from Open-Meteo API
  4. Clean and transform raw payloads
  5. Write results to RDS PostgreSQL via bulk insert
  6. Log ETL run audit records
  7. Refresh materialized views on completion

Designed to run as:
  - A direct Python script (cron or subprocess)
  - Inside an AWS Lambda handler wrapper
  - Via boto3 orchestrator invocation
"""

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from config.logging_config import get_logger
from config.settings import etl as etl_cfg
from database.db_manager import db_manager

logger = get_logger(__name__, log_file="/var/log/areca/etl_pipeline.log")

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PriceRecord:
    record_date: date
    market_name: str
    district: str
    variety_name: str
    min_price: float
    max_price: float
    modal_price: float
    arrivals_tons: Optional[float]
    source: str
    raw_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeatherRecord:
    record_date: date
    market_name: str
    rainfall_mm: Optional[float]
    avg_humidity: Optional[float]
    temperature_c: Optional[float]
    wind_speed_kmh: Optional[float]
    source: str = "open-meteo"


# ---------------------------------------------------------------------------
# HTTP Session with retry logic
# ---------------------------------------------------------------------------


class RobustHTTPSession:
    """HTTP session with exponential backoff retries and browser-like headers."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9,kn;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def get(
        self,
        url: str,
        params: Optional[Dict] = None,
        timeout: int = etl_cfg.scrape_timeout_sec,
        retries: int = etl_cfg.max_retries,
    ) -> Optional[requests.Response]:
        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else "N/A"
                logger.warning(
                    "HTTP error on attempt",
                    attempt=attempt,
                    url=url,
                    status=status,
                )
                if status == 429:
                    time.sleep(60)  # Rate-limited: wait 60 seconds
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                logger.warning(
                    "Connection/timeout error on attempt",
                    attempt=attempt,
                    url=url,
                    error=str(exc),
                )

            if attempt < retries:
                sleep_time = etl_cfg.retry_delay_sec * (2 ** (attempt - 1))
                logger.info("Retrying after backoff", sleep_seconds=sleep_time)
                time.sleep(sleep_time)

        logger.error("All retry attempts exhausted", url=url)
        return None

    def post(
        self,
        url: str,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: int = etl_cfg.scrape_timeout_sec,
        retries: int = etl_cfg.max_retries,
    ) -> Optional[requests.Response]:
        extra_headers = headers or {}
        for attempt in range(1, retries + 1):
            try:
                response = self.session.post(
                    url, data=data, headers={**self.HEADERS, **extra_headers}, timeout=timeout
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "POST error on attempt", attempt=attempt, url=url, error=str(exc)
                )
                if attempt < retries:
                    time.sleep(etl_cfg.retry_delay_sec * attempt)

        logger.error("POST retry attempts exhausted", url=url)
        return None


# ---------------------------------------------------------------------------
# Agmarknet Scraper
# ---------------------------------------------------------------------------


class AgmarknetScraper:
    """
    Scrapes daily areca nut price data from agmarknet.gov.in.

    Agmarknet uses ASP.NET WebForms with ViewState. We:
      1. GET the search page to extract ViewState tokens
      2. POST the form with commodity/market/state filters
      3. Parse the resulting HTML price table
    """

    BASE_URL = etl_cfg.agmarknet_base_url
    SEARCH_URL = etl_cfg.agmarknet_price_url
    COMMODITY_CODE = "23"  # Areca Nut
    STATE_CODE = "Karnataka"

    def __init__(self, http: RobustHTTPSession):
        self.http = http

    def _extract_viewstate(self, html: str) -> Dict[str, str]:
        """Extract ASP.NET hidden form fields required for POST."""
        soup = BeautifulSoup(html, "html.parser")
        fields = {}
        for field_name in [
            "__VIEWSTATE",
            "__VIEWSTATEGENERATOR",
            "__EVENTVALIDATION",
            "__EVENTTARGET",
            "__EVENTARGUMENT",
        ]:
            tag = soup.find("input", {"name": field_name})
            if tag:
                fields[field_name] = tag.get("value", "")
        return fields

    def _build_form_payload(
        self,
        viewstate_fields: Dict[str, str],
        target_date: date,
    ) -> Dict[str, str]:
        """Construct the form POST payload for Agmarknet search."""
        date_str = target_date.strftime("%d-%b-%Y")
        return {
            **viewstate_fields,
            "ctl00$MainContent$Ddl_Commodity": self.COMMODITY_CODE,
            "ctl00$MainContent$Ddl_State": self.STATE_CODE,
            "ctl00$MainContent$Ddl_District": "0",  # All districts
            "ctl00$MainContent$Ddl_Market": "0",    # All markets
            "ctl00$MainContent$txt_Date": date_str,
            "ctl00$MainContent$btn_search": "Search",
            "__EVENTTARGET": "",
            "__EVENTARGUMENT": "",
        }

    def _parse_price_table(
        self, html: str, target_date: date, source: str = "agmarknet"
    ) -> List[PriceRecord]:
        """Parse the HTML price table returned by Agmarknet search."""
        soup = BeautifulSoup(html, "html.parser")
        records: List[PriceRecord] = []

        # Agmarknet renders the data in a table with id 'GridView_PriceReport'
        table = soup.find("table", {"id": "GridView_PriceReport"})
        if table is None:
            # Fallback: try any table with 'Modal Price' header
            tables = soup.find_all("table")
            for t in tables:
                headers = [th.get_text(strip=True).lower() for th in t.find_all("th")]
                if "modal price" in headers or "modal_price" in " ".join(headers):
                    table = t
                    break

        if table is None:
            logger.warning("Price table not found in Agmarknet response", date=str(target_date))
            return records

        rows = table.find_all("tr")
        if len(rows) < 2:
            return records

        # Parse header to map column positions
        header_row = rows[0]
        headers = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]

        col_map = {}
        for idx, h in enumerate(headers):
            if "state" in h:
                col_map["state"] = idx
            elif "district" in h:
                col_map["district"] = idx
            elif "market" in h:
                col_map["market"] = idx
            elif "commodity" in h:
                col_map["commodity"] = idx
            elif "variety" in h:
                col_map["variety"] = idx
            elif "grade" in h:
                col_map["grade"] = idx
            elif "min" in h:
                col_map["min_price"] = idx
            elif "max" in h:
                col_map["max_price"] = idx
            elif "modal" in h:
                col_map["modal_price"] = idx
            elif "arrival" in h:
                col_map["arrivals"] = idx

        for row in rows[1:]:
            cells = row.find_all("td")
            if not cells:
                continue

            def cell(key: str, default: str = "") -> str:
                idx = col_map.get(key)
                if idx is not None and idx < len(cells):
                    return cells[idx].get_text(strip=True)
                return default

            market_name = cell("market")
            district = cell("district", "Unknown")
            variety_raw = cell("variety") or cell("grade", "Chali")

            # Normalize variety to our known list
            variety = self._normalize_variety(variety_raw)
            if not variety:
                continue  # skip unknown varieties for areca nut

            try:
                min_price = self._parse_price(cell("min_price"))
                max_price = self._parse_price(cell("max_price"))
                modal_price = self._parse_price(cell("modal_price"))

                # Basic sanity checks
                if modal_price <= 0 or min_price > max_price:
                    continue
                if modal_price < min_price or modal_price > max_price:
                    modal_price = (min_price + max_price) / 2

                arrivals_raw = cell("arrivals")
                arrivals_tons = self._parse_float(arrivals_raw)

                records.append(
                    PriceRecord(
                        record_date=target_date,
                        market_name=market_name.strip(),
                        district=district.strip(),
                        variety_name=variety,
                        min_price=round(min_price, 2),
                        max_price=round(max_price, 2),
                        modal_price=round(modal_price, 2),
                        arrivals_tons=arrivals_tons,
                        source=source,
                        raw_payload={
                            "raw_variety": variety_raw,
                            "raw_arrivals": arrivals_raw,
                        },
                    )
                )
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping malformed row", error=str(exc))
                continue

        logger.info("Agmarknet rows parsed", count=len(records), date=str(target_date))
        return records

    @staticmethod
    def _normalize_variety(raw: str) -> Optional[str]:
        """Map raw Agmarknet variety strings to our canonical variety names."""
        mapping = {
            "chali": "Chali",
            "supari": "Chali",
            "betelnut": "Chali",
            "gotu": "Gotu",
            "whole": "Gotu",
            "raw": "Gotu",
            "kotte": "Kotte",
            "tender": "Kotte",
            "rashi": "Rashi",
            "mixed": "Rashi",
            "saraku": "Saraku",
            "processed": "Saraku",
        }
        normalized = raw.strip().lower()
        for key, value in mapping.items():
            if key in normalized:
                return value
        # If no match found and it contains 'areca', default to Chali
        if "areca" in normalized:
            return "Chali"
        return None

    @staticmethod
    def _parse_price(raw: str) -> float:
        cleaned = re.sub(r"[^\d.]", "", raw.replace(",", ""))
        return float(cleaned) if cleaned else 0.0

    @staticmethod
    def _parse_float(raw: str) -> Optional[float]:
        cleaned = re.sub(r"[^\d.]", "", raw.replace(",", ""))
        return float(cleaned) if cleaned else None

    def scrape_date(self, target_date: date) -> List[PriceRecord]:
        """Scrape all areca nut prices for a specific date."""
        logger.info("Scraping Agmarknet", date=str(target_date))

        # Step 1: GET the search page for ViewState tokens
        response = self.http.get(self.SEARCH_URL, timeout=45)
        if not response:
            logger.error("Failed to load Agmarknet search page")
            return []

        viewstate = self._extract_viewstate(response.text)
        if not viewstate.get("__VIEWSTATE"):
            logger.warning("ViewState token missing - page structure may have changed")

        # Step 2: POST the search form
        payload = self._build_form_payload(viewstate, target_date)
        post_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": self.SEARCH_URL,
            "Origin": self.BASE_URL,
        }
        result_response = self.http.post(
            self.SEARCH_URL, data=payload, headers=post_headers, timeout=60
        )

        if not result_response:
            logger.error("Failed to get Agmarknet search results")
            return []

        return self._parse_price_table(result_response.text, target_date)

    def scrape_date_range(
        self, start_date: date, end_date: date
    ) -> List[PriceRecord]:
        """Scrape prices for a date range (inclusive)."""
        all_records: List[PriceRecord] = []
        current = start_date
        while current <= end_date:
            records = self.scrape_date(current)
            all_records.extend(records)
            # Be respectful to the server
            time.sleep(2.5)
            current += timedelta(days=1)
        logger.info(
            "Date range scrape complete",
            start=str(start_date),
            end=str(end_date),
            total_records=len(all_records),
        )
        return all_records


# ---------------------------------------------------------------------------
# APMC Regional Portal Scraper (Secondary / Fallback)
# ---------------------------------------------------------------------------


class APMCScraper:
    """
    Scrapes Karnataka APMC digital portal (e-market data).
    This serves as a fallback and cross-validation source.
    Target: https://www.krishimaratavahini.kar.nic.in (Karnataka e-market portal)
    """

    PORTAL_URL = "https://www.krishimaratavahini.kar.nic.in/marketInfo.aspx"
    COMMODITY_ID = "23"  # Areca nut

    def __init__(self, http: RobustHTTPSession):
        self.http = http

    def _parse_market_table(self, html: str, target_date: date) -> List[PriceRecord]:
        """Parse the APMC portal HTML table."""
        soup = BeautifulSoup(html, "html.parser")
        records: List[PriceRecord] = []

        # The APMC portal uses a standard Bootstrap table
        tables = soup.find_all("table", class_=re.compile(r"table", re.I))
        if not tables:
            tables = soup.find_all("table")

        for table in tables:
            headers_raw = [
                th.get_text(strip=True).lower()
                for th in table.find_all("th")
            ]
            if not any("price" in h for h in headers_raw):
                continue

            for row in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) < 5:
                    continue
                try:
                    market_name = cells[0]
                    variety_name = AgmarknetScraper._normalize_variety(cells[1])
                    if not variety_name:
                        continue
                    min_price = AgmarknetScraper._parse_price(cells[2])
                    max_price = AgmarknetScraper._parse_price(cells[3])
                    modal_price = AgmarknetScraper._parse_price(cells[4])
                    arrivals = AgmarknetScraper._parse_float(cells[5]) if len(cells) > 5 else None

                    if modal_price <= 0:
                        continue

                    records.append(
                        PriceRecord(
                            record_date=target_date,
                            market_name=market_name,
                            district="Karnataka",
                            variety_name=variety_name,
                            min_price=round(min_price, 2),
                            max_price=round(max_price, 2),
                            modal_price=round(modal_price, 2),
                            arrivals_tons=arrivals,
                            source="apmc-karnataka",
                            raw_payload={"raw_cells": cells},
                        )
                    )
                except Exception as exc:
                    logger.debug("APMC row parse error", error=str(exc))

        logger.info("APMC records parsed", count=len(records), date=str(target_date))
        return records

    def scrape_date(self, target_date: date) -> List[PriceRecord]:
        """Scrape APMC portal for a given date."""
        params = {
            "commodityId": self.COMMODITY_ID,
            "date": target_date.strftime("%d/%m/%Y"),
        }
        response = self.http.get(self.PORTAL_URL, params=params, timeout=30)
        if not response:
            logger.warning("APMC portal unreachable", date=str(target_date))
            return []
        return self._parse_market_table(response.text, target_date)


# ---------------------------------------------------------------------------
# Weather Data Fetcher
# ---------------------------------------------------------------------------


class WeatherFetcher:
    """
    Fetches historical weather data for areca nut growing regions
    using the Open-Meteo API (free, no API key required for historical data).
    """

    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = etl_cfg.weather_api_url

    def __init__(self, http: RobustHTTPSession):
        self.http = http

    def fetch_historical(
        self, market_name: str, target_date: date
    ) -> Optional[WeatherRecord]:
        """Fetch historical weather for a specific market and date."""
        location = etl_cfg.weather_locations.get(market_name)
        if not location:
            logger.debug("No weather location config", market=market_name)
            return None

        params = {
            "latitude": location["lat"],
            "longitude": location["lon"],
            "start_date": str(target_date),
            "end_date": str(target_date),
            "daily": "precipitation_sum,relative_humidity_2m_mean,temperature_2m_mean,wind_speed_10m_max",
            "timezone": "Asia/Kolkata",
        }

        response = self.http.get(self.HISTORICAL_URL, params=params, timeout=20)
        if not response:
            logger.warning("Weather API call failed", market=market_name, date=str(target_date))
            return None

        try:
            data = response.json()
            daily = data.get("daily", {})
            if not daily.get("time") or not daily["time"]:
                return None

            return WeatherRecord(
                record_date=target_date,
                market_name=market_name,
                rainfall_mm=self._safe_float(
                    daily.get("precipitation_sum", [None])[0]
                ),
                avg_humidity=self._safe_float(
                    daily.get("relative_humidity_2m_mean", [None])[0]
                ),
                temperature_c=self._safe_float(
                    daily.get("temperature_2m_mean", [None])[0]
                ),
                wind_speed_kmh=self._safe_float(
                    daily.get("wind_speed_10m_max", [None])[0]
                ),
                source="open-meteo-archive",
            )
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("Weather data parse error", error=str(exc))
            return None

    def fetch_all_markets(self, target_date: date) -> List[WeatherRecord]:
        """Fetch weather for all configured areca nut markets."""
        results: List[WeatherRecord] = []
        for market_name in etl_cfg.weather_locations.keys():
            record = self.fetch_historical(market_name, target_date)
            if record:
                results.append(record)
            time.sleep(0.5)  # Respect Open-Meteo rate limits
        logger.info("Weather records fetched", count=len(results), date=str(target_date))
        return results

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Data Transformer
# ---------------------------------------------------------------------------


class DataTransformer:
    """Validates, deduplicates, and normalizes raw scraped data before DB write."""

    @staticmethod
    def validate_price_record(record: PriceRecord) -> bool:
        """Return True if the price record passes all validation checks."""
        if not record.market_name or not record.variety_name:
            return False
        if record.min_price < 0 or record.max_price < 0 or record.modal_price < 0:
            return False
        if record.min_price > record.max_price:
            return False
        # Sanity range for areca nut prices in INR per quintal
        if record.modal_price < 10_000 or record.modal_price > 2_000_000:
            logger.debug(
                "Price out of expected range — skipping",
                market=record.market_name,
                variety=record.variety_name,
                modal=record.modal_price,
            )
            return False
        return True

    @staticmethod
    def deduplicate(records: List[PriceRecord]) -> List[PriceRecord]:
        """Remove duplicate records based on (date, market, variety) key."""
        seen = set()
        unique: List[PriceRecord] = []
        for rec in records:
            key = (str(rec.record_date), rec.market_name.lower(), rec.variety_name.lower())
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        return unique

    @staticmethod
    def merge_sources(
        primary: List[PriceRecord], secondary: List[PriceRecord]
    ) -> List[PriceRecord]:
        """
        Merge primary (Agmarknet) and secondary (APMC) records.
        Primary records take precedence for the same (date, market, variety).
        """
        primary_keys = {
            (str(r.record_date), r.market_name.lower(), r.variety_name.lower())
            for r in primary
        }
        supplemental = [
            r for r in secondary
            if (str(r.record_date), r.market_name.lower(), r.variety_name.lower())
            not in primary_keys
        ]
        return primary + supplemental


# ---------------------------------------------------------------------------
# Database Writer
# ---------------------------------------------------------------------------


class ETLDatabaseWriter:
    """Handles resolving market/variety IDs and writing records to PostgreSQL."""

    def __init__(self):
        self._market_cache: Dict[str, str] = {}   # market_name -> market_id (UUID)
        self._variety_cache: Dict[str, int] = {}  # variety_name -> variety_id (int)

    def load_caches(self) -> None:
        """Pre-load market and variety ID lookups from DB."""
        markets = db_manager.execute_query(
            "SELECT market_id::text, market_name FROM markets WHERE active = TRUE"
        )
        for row in markets:
            self._market_cache[row["market_name"].lower()] = row["market_id"]

        varieties = db_manager.execute_query(
            "SELECT variety_id, variety_name FROM varieties WHERE active = TRUE"
        )
        for row in varieties:
            self._variety_cache[row["variety_name"].lower()] = row["variety_id"]

        logger.info(
            "Caches loaded",
            markets=len(self._market_cache),
            varieties=len(self._variety_cache),
        )

    def _get_or_create_market(
        self, market_name: str, district: str
    ) -> Optional[str]:
        """Return market_id, creating the market record if it doesn't exist."""
        key = market_name.lower()
        if key in self._market_cache:
            return self._market_cache[key]

        # Insert new market
        rows = db_manager.execute_query(
            """
            INSERT INTO markets (market_name, district, state)
            VALUES (%s, %s, 'Karnataka')
            ON CONFLICT (market_name, district) DO UPDATE SET active = TRUE
            RETURNING market_id::text
            """,
            (market_name, district),
        )
        if rows:
            market_id = rows[0]["market_id"]
            self._market_cache[key] = market_id
            logger.info("Created new market", market=market_name, id=market_id)
            return market_id

        return None

    def write_prices(self, records: List[PriceRecord]) -> Tuple[int, int]:
        """
        Write price records to market_prices table.

        Returns:
            Tuple of (rows_written, rows_skipped)
        """
        rows_written = 0
        rows_skipped = 0
        rows: List[Tuple] = []

        for rec in records:
            market_id = self._get_or_create_market(rec.market_name, rec.district)
            variety_id = self._variety_cache.get(rec.variety_name.lower())

            if not market_id or not variety_id:
                logger.debug(
                    "Skipping record — unknown market or variety",
                    market=rec.market_name,
                    variety=rec.variety_name,
                )
                rows_skipped += 1
                continue

            rows.append((
                rec.record_date,
                market_id,
                variety_id,
                rec.min_price,
                rec.max_price,
                rec.modal_price,
                rec.arrivals_tons,
                rec.source,
                json.dumps(rec.raw_payload),
            ))

        if rows:
            inserted = db_manager.bulk_insert(
                table="market_prices",
                columns=[
                    "record_date", "market_id", "variety_id",
                    "min_price", "max_price", "modal_price",
                    "arrivals_tons", "source", "raw_payload",
                ],
                rows=rows,
                on_conflict=(
                    "(record_date, market_id, variety_id) "
                    "DO UPDATE SET "
                    "  min_price    = EXCLUDED.min_price, "
                    "  max_price    = EXCLUDED.max_price, "
                    "  modal_price  = EXCLUDED.modal_price, "
                    "  arrivals_tons = EXCLUDED.arrivals_tons, "
                    "  ingested_at  = NOW()"
                ),
            )
            rows_written = len(rows)
            logger.info("Price records written", written=rows_written, skipped=rows_skipped)

        return rows_written, rows_skipped

    def write_weather(self, records: List[WeatherRecord]) -> int:
        """Write weather records to weather_metrics table."""
        rows_written = 0
        rows: List[Tuple] = []

        for rec in records:
            market_id = self._market_cache.get(rec.market_name.lower())
            if not market_id:
                logger.debug("Skipping weather — unknown market", market=rec.market_name)
                continue

            rows.append((
                rec.record_date,
                market_id,
                rec.rainfall_mm,
                rec.avg_humidity,
                rec.temperature_c,
                rec.wind_speed_kmh,
                rec.source,
            ))

        if rows:
            db_manager.bulk_insert(
                table="weather_metrics",
                columns=[
                    "record_date", "region_id", "rainfall_mm",
                    "avg_humidity", "temperature_c", "wind_speed_kmh", "source",
                ],
                rows=rows,
                on_conflict=(
                    "(record_date, region_id) "
                    "DO UPDATE SET "
                    "  rainfall_mm   = EXCLUDED.rainfall_mm, "
                    "  avg_humidity  = EXCLUDED.avg_humidity, "
                    "  temperature_c = EXCLUDED.temperature_c, "
                    "  wind_speed_kmh = EXCLUDED.wind_speed_kmh"
                ),
            )
            rows_written = len(rows)
            logger.info("Weather records written", count=rows_written)

        return rows_written


# ---------------------------------------------------------------------------
# ETL Audit Logger
# ---------------------------------------------------------------------------


def log_etl_run(
    source: str,
    status: str,
    records_fetched: int = 0,
    records_written: int = 0,
    records_skipped: int = 0,
    error_message: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """Insert/update an ETL run audit record."""
    try:
        db_manager.execute_query(
            """
            INSERT INTO etl_runs
                (source, status, records_fetched, records_written, records_skipped,
                 completed_at, error_message, metadata)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s)
            """,
            (
                source,
                status,
                records_fetched,
                records_written,
                records_skipped,
                error_message,
                json.dumps(metadata or {}),
            ),
        )
    except Exception as exc:
        logger.error("Failed to log ETL run", error=str(exc))


# ---------------------------------------------------------------------------
# Main ETL Orchestrator
# ---------------------------------------------------------------------------


class ETLOrchestrator:
    """
    Top-level ETL coordinator that:
    1. Determines the date(s) to process
    2. Runs Agmarknet scraper → APMC fallback scraper
    3. Fetches weather data
    4. Validates, deduplicates, and merges records
    5. Writes to RDS
    6. Refreshes materialized views
    7. Logs audit records
    """

    def __init__(self):
        self.http = RobustHTTPSession()
        self.agmarknet = AgmarknetScraper(self.http)
        self.apmc = APMCScraper(self.http)
        self.weather = WeatherFetcher(self.http)
        self.transformer = DataTransformer()
        self.writer = ETLDatabaseWriter()

    def run(
        self,
        target_date: Optional[date] = None,
        backfill_days: int = 0,
    ) -> Dict[str, Any]:
        """
        Execute the full ETL pipeline.

        Args:
            target_date:   Date to process (defaults to yesterday for stable data).
            backfill_days: If > 0, process the last N days in addition.

        Returns:
            Summary dict with run statistics.
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        dates_to_process = [target_date - timedelta(days=i) for i in range(backfill_days + 1)]
        dates_to_process = sorted(set(dates_to_process))  # deduplicate, ascending

        logger.info(
            "ETL run starting",
            dates=str(dates_to_process),
            count=len(dates_to_process),
        )

        db_manager.initialize()
        self.writer.load_caches()

        total_fetched = 0
        total_written = 0
        total_skipped = 0
        total_weather = 0
        errors: List[str] = []

        for process_date in dates_to_process:
            try:
                # --- Price Scraping ---
                primary_records = self.agmarknet.scrape_date(process_date)
                time.sleep(3)  # Rate limit courtesy delay

                secondary_records = []
                if len(primary_records) < 5:
                    logger.info(
                        "Low primary count — triggering APMC fallback",
                        primary_count=len(primary_records),
                        date=str(process_date),
                    )
                    secondary_records = self.apmc.scrape_date(process_date)

                # Merge and validate
                merged = self.transformer.merge_sources(primary_records, secondary_records)
                validated = [r for r in merged if self.transformer.validate_price_record(r)]
                deduplicated = self.transformer.deduplicate(validated)

                fetched = len(deduplicated)
                total_fetched += fetched

                # Write prices
                written, skipped = self.writer.write_prices(deduplicated)
                total_written += written
                total_skipped += skipped

                # --- Weather Fetching ---
                weather_records = self.weather.fetch_all_markets(process_date)
                total_weather += self.writer.write_weather(weather_records)

            except Exception as exc:
                error_msg = f"{process_date}: {type(exc).__name__}: {exc}"
                logger.error("Error processing date", error=error_msg, exc_info=True)
                errors.append(error_msg)

        # Refresh materialized views
        try:
            db_manager.refresh_views()
        except Exception as exc:
            logger.error("View refresh failed", error=str(exc))

        status = "success" if not errors else ("partial" if total_written > 0 else "failed")
        summary = {
            "status": status,
            "dates_processed": [str(d) for d in dates_to_process],
            "records_fetched": total_fetched,
            "records_written": total_written,
            "records_skipped": total_skipped,
            "weather_records_written": total_weather,
            "errors": errors,
        }

        log_etl_run(
            source="agmarknet+apmc+weather",
            status=status,
            records_fetched=total_fetched,
            records_written=total_written,
            records_skipped=total_skipped,
            error_message="; ".join(errors) if errors else None,
            metadata={"dates": [str(d) for d in dates_to_process]},
        )

        logger.info("ETL run complete", **summary)
        return summary


# ---------------------------------------------------------------------------
# Lambda Handler Entry Point
# ---------------------------------------------------------------------------


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda entry point for ETL pipeline.
    Triggered by CloudWatch Events / EventBridge scheduled rule.
    """
    logger.info("Lambda ETL handler invoked", event=event)

    target_date_str = event.get("target_date")
    backfill_days = int(event.get("backfill_days", 0))

    target_date: Optional[date] = None
    if target_date_str:
        try:
            target_date = date.fromisoformat(target_date_str)
        except ValueError:
            logger.error("Invalid target_date in event", value=target_date_str)

    orchestrator = ETLOrchestrator()
    summary = orchestrator.run(target_date=target_date, backfill_days=backfill_days)

    return {
        "statusCode": 200 if summary["status"] in ("success", "partial") else 500,
        "body": json.dumps(summary),
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Areca Nut ETL Pipeline")
    parser.add_argument(
        "--date",
        type=str,
        help="Target date in YYYY-MM-DD format (default: yesterday)",
        default=None,
    )
    parser.add_argument(
        "--backfill",
        type=int,
        help="Number of additional days to backfill",
        default=0,
    )
    args = parser.parse_args()

    target: Optional[date] = None
    if args.date:
        target = date.fromisoformat(args.date)

    orchestrator = ETLOrchestrator()
    result = orchestrator.run(target_date=target, backfill_days=args.backfill)
    print(json.dumps(result, indent=2))
