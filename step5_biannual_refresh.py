"""
step5_biannual_refresh.py
──────────────────────────
Standalone script — NOT a background thread inside app.py.

This is meant to be run by Render's Cron Jobs feature (or any external
scheduler / `cron` / Task Scheduler) on a schedule like "every 6 months."
It does NOT keep running in the background — it starts, loads the
ontology, checks every manufacturer's live privacy policy against the
KG, updates anything that's stale, writes the ontology back to disk,
logs what happened, and exits.

Why standalone instead of a thread in app.py:
  - Render web services sleep/restart independently of any in-memory
    timer you start inside Flask. A thread-based scheduler resets to
    zero on every restart/deploy and never reaches "6 months."
  - Cron Jobs are a separate Render resource that Render itself wakes
    up on schedule, runs to completion, then tears down. No dependency
    on your web service being awake.
  - Running as a short-lived process also means no risk of two workers
    each starting their own copy of a long-running thread and stepping
    on the same rdflib Graph / ontology file at the same time.

Usage (local test):
    python step5_biannual_refresh.py --run
    python step5_biannual_refresh.py --run --dry-run

Render Cron Job command:
    python step5_biannual_refresh.py --run

Render Cron Job schedule (every 6 months, midnight on the 1st):
    0 0 1 */6 *
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Graph, URIRef, RDFS, Literal
from groq import Groq

import step4a_timestamps as ts
import agentic_policy_update_crawler as policy_crawler

# A .env file sitting on disk does NOT automatically populate os.environ —
# something has to explicitly load it. python-dotenv does that here so
# GROQ_API_KEY / BRAVE_SEARCH_API_KEY from your .env are actually visible
# to os.environ.get(...) below. If python-dotenv isn't installed, this
# script still runs — it just relies on real environment variables instead
# (which is exactly what happens on Render, since Render injects env vars
# directly rather than via a .env file).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("biannual_refresh")

REFRESH_LOG_PATH = "biannual_refresh_log.json"
LAST_RUN_PATH = "biannual_refresh_last_run.json"


# ---------------------------------------------------------------------------
# Minimal, standalone KG loading — mirrors the startup block in app.py but
# does NOT import app.py (importing app.py would try to start Flask, build
# the BERT index, etc., which is unnecessary work for a cron job that only
# needs the graph + manufacturer list + policy property).
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_graph(cfg: dict) -> tuple[Graph, Path]:
    onto_path = Path(os.environ.get("ONTO_PATH", "") or cfg["ontology_path"]).resolve()
    g = Graph()
    g.parse(str(onto_path))
    logger.info("Loaded ontology: %s (%d triples)", onto_path, len(g))
    return g, onto_path


_NAME_FIX = {"adt": "ADT"}


def clean_name(n: str) -> str:
    return _NAME_FIX.get(n.lower().strip(), n.title())


def local_name(iri: str) -> str:
    import re
    s = str(iri).rsplit("#", 1)[-1].rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s).replace("_", " ").replace("-", " ")


def load_manufacturers(g: Graph, cfg: dict) -> list[dict]:
    """Same query app.py uses at startup — kept independent on purpose."""
    mfg_cls = URIRef(cfg["manufacturer_class_iri"])
    policy_prop = URIRef(cfg["policy_property_iri"])

    manufacturers = []
    q = f"SELECT DISTINCT ?i WHERE {{ ?i a ?t . ?t <{RDFS.subClassOf}>* <{mfg_cls}> . }}"
    for row in g.query(q):
        inst = row.i
        lbls = [str(o) for o in g.objects(inst, RDFS.label)]
        name = clean_name(lbls[0] if lbls else local_name(str(inst)))
        pols = [str(o) for o in g.objects(inst, policy_prop) if isinstance(o, Literal)]
        pol = max(pols, key=len) if pols else ""
        if not pol.strip():
            continue  # nothing to refresh-check for manufacturers with no stored policy yet
        manufacturers.append({"iri": str(inst), "name": name, "policy": pol})

    manufacturers.sort(key=lambda m: m["name"].lower())
    logger.info("Loaded %d manufacturers with stored policies", len(manufacturers))
    return manufacturers


def make_groq_client() -> Groq | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — date extraction will fall back to regex-only, "
                        "and URL discovery will fall back to highest-scored search result.")
        return None
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Logging helpers — append-only audit trail + "when did we last run" marker.
# The last-run marker matters because Cron Jobs are stateless between
# invocations; without writing this to disk (or a DB) there is no way to
# tell, from inside the script, whether the schedule is actually being
# honored by the external scheduler.
# ---------------------------------------------------------------------------

def _read_json(path: str, default):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Could not read %s, starting fresh", path)
        return default


def _append_refresh_log(entry: dict):
    log = _read_json(REFRESH_LOG_PATH, [])
    log.append(entry)
    Path(REFRESH_LOG_PATH).write_text(json.dumps(log, indent=2), encoding="utf-8")


def _write_last_run_marker(summary: dict):
    Path(LAST_RUN_PATH).write_text(json.dumps({
        "last_run_at": summary["finished_at"],
        "dry_run": summary["dry_run"],
        "updated": summary["updated"],
        "errors": summary["errors"],
    }, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core sweep — checks every manufacturer once, updates the KG where stale.
# ---------------------------------------------------------------------------

def run_biannual_refresh(
    g: Graph,
    manufacturers: list[dict],
    policy_prop: URIRef,
    onto_path: str,
    groq_client=None,
    dry_run: bool = False,
    sleep_between: float = 2.0,
) -> dict:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []

    for mfg in manufacturers:
        name = mfg["name"]
        iri = URIRef(mfg["iri"])
        try:
            if dry_run:
                source = policy_crawler.discover_policy_url_agent(
                    company_name=name, llm_client=groq_client)
                policy_url = source.get("selected_policy_url")
                if not policy_url:
                    results.append({"company": name, "status": "source_not_found"})
                    continue

                fetched = policy_crawler.fetch_policy_source(policy_url)
                live_text = policy_crawler.extract_policy_text(fetched)
                live_date, _ = policy_crawler.extract_policy_date_agent(
                    live_text, fetched.get("last_modified_header"), llm_client=groq_client)
                stored_date = policy_crawler.get_stored_company_date(g, iri)
                decision = policy_crawler.compare_policy_dates(stored_date, live_date)

                results.append({
                    "company": name,
                    "status": decision["status"],
                    "would_update": decision["should_update"],
                    "policy_url": policy_url,
                })
            else:
                report = policy_crawler.auto_check_and_update_company_policy(
                    company_name=name,
                    manufacturer_iri=iri,
                    stored_policy_text=mfg.get("policy", ""),
                    g=g,
                    policy_prop=policy_prop,
                    onto_path=onto_path,
                    groq_client=groq_client,
                    timestamp_module=ts,
                )
                results.append({
                    "company": name,
                    "status": report.get("status"),
                    "action_taken": report.get("action_taken"),
                    "discovered_policy_url": report.get("discovered_policy_url"),
                })

            logger.info("Checked %s -> %s", name, results[-1].get("status"))

        except Exception as e:
            logger.exception("Refresh failed for %s", name)
            results.append({"company": name, "status": "error", "error": str(e)})

        time.sleep(sleep_between)  # be polite to target servers / search API

    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {
        "started_at": started,
        "finished_at": finished,
        "dry_run": dry_run,
        "total_checked": len(results),
        "updated": sum(1 for r in results if r.get("status") == "updated"),
        "current": sum(1 for r in results if r.get("status") == "current"),
        "errors": sum(1 for r in results if r.get("status") == "error"),
        "results": results,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI entry point — this is what Render's Cron Job actually invokes.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Biannual KG privacy-policy refresh sweep.")
    ap.add_argument("--run", action="store_true", help="Actually run the sweep (required, safety guard).")
    ap.add_argument("--dry-run", action="store_true", help="Check dates only; do not write to the KG.")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--sleep-between", type=float, default=2.0,
                     help="Seconds to wait between manufacturers (politeness delay).")
    args = ap.parse_args()

    if not args.run:
        ap.error("Refusing to run without --run (safety guard against accidental invocation).")

    cfg = load_config(args.config)
    g, onto_path = load_graph(cfg)
    policy_prop = URIRef(cfg["policy_property_iri"])
    manufacturers = load_manufacturers(g, cfg)
    groq_client = make_groq_client()

    if not manufacturers:
        logger.warning("No manufacturers with stored policies found — nothing to refresh.")
        sys.exit(0)

    summary = run_biannual_refresh(
        g=g,
        manufacturers=manufacturers,
        policy_prop=policy_prop,
        onto_path=str(onto_path),
        groq_client=groq_client,
        dry_run=args.dry_run,
        sleep_between=args.sleep_between,
    )

    if not args.dry_run:
        g.serialize(destination=str(onto_path), format="xml")
        logger.info("Ontology re-serialized to %s", onto_path)

    _append_refresh_log(summary)
    _write_last_run_marker(summary)

    logger.info(
        "Sweep complete: %d checked, %d updated, %d current, %d errors",
        summary["total_checked"], summary["updated"], summary["current"], summary["errors"],
    )

    # Non-zero exit if everything errored, so Render's Cron Job dashboard
    # surfaces a failed run instead of a silent green checkmark.
    if summary["errors"] and summary["errors"] == summary["total_checked"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
