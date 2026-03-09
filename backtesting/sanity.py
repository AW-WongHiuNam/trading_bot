from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import json


FORBIDDEN_PRICE_FUNCTIONS = {
    "GLOBAL_QUOTE",
    "TIME_SERIES_DAILY",
    "TIME_SERIES_INTRADAY",
    "TIME_SERIES_WEEKLY",
    "TIME_SERIES_MONTHLY",
}

ALLOWED_NEWS_FUNCTIONS = {
    "NEWS_SENTIMENT",
}


@dataclass
class SanityReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.passed = False
        self.errors.append(message)


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def check_required_fields(result: dict, report: SanityReport) -> None:
    required = ["trader_proposal", "manager_decision", "risk", "trace"]
    for key in required:
        if key not in result:
            report.fail(f"Missing required field: {key}")


def check_no_future_timestamps(result: dict, cutoff_date: date, report: SanityReport, *, enforce_runtime_cutoff: bool) -> None:
    latest_allowed = datetime.combine(cutoff_date, datetime.max.time())

    ts_candidates: list[tuple[str, str | None]] = [
        ("manager_decision.timestamp", (result.get("manager_decision") or {}).get("timestamp")),
        ("trader_proposal.timestamp", (result.get("trader_proposal") or {}).get("timestamp")),
    ]

    analysts = result.get("analysts") or {}
    if isinstance(analysts, dict):
        for name, payload in analysts.items():
            stamp = payload.get("timestamp") if isinstance(payload, dict) else None
            ts_candidates.append((f"analysts.{name}.timestamp", stamp))

    trace = result.get("trace") or []
    if isinstance(trace, list):
        for idx, event in enumerate(trace):
            stamp = event.get("at") if isinstance(event, dict) else None
            ts_candidates.append((f"trace[{idx}].at", stamp))

    for label, raw in ts_candidates:
        parsed = _parse_iso_dt(raw)
        if parsed is None:
            continue
        if parsed > latest_allowed:
            message = f"Runtime timestamp after cutoff: {label}={raw} > {cutoff_date.isoformat()}"
            if enforce_runtime_cutoff:
                report.fail(message)
            else:
                report.warnings.append(message)


def _try_parse_json_preview(preview: str | None) -> dict | None:
    if not isinstance(preview, str) or not preview.strip():
        return None
    try:
        return json.loads(preview)
    except Exception:
        return None


def check_no_future_data_dates(result: dict, cutoff_date: date, report: SanityReport) -> None:
    trace = result.get("trace") or []
    if not isinstance(trace, list):
        return

    for idx, event in enumerate(trace):
        if not isinstance(event, dict):
            continue
        function = str(event.get("function") or "").upper()
        parsed = _try_parse_json_preview(event.get("response_preview"))
        if not parsed:
            continue

        if function == "GLOBAL_QUOTE":
            latest = ((parsed.get("Global Quote") or {}).get("07. latest trading day"))
            if isinstance(latest, str):
                try:
                    market_date = datetime.strptime(latest, "%Y-%m-%d").date()
                    if market_date > cutoff_date:
                        report.fail(f"Future market date in trace[{idx}] GLOBAL_QUOTE: {market_date}")
                except Exception:
                    pass

        if function == "NEWS_SENTIMENT":
            feed = parsed.get("feed") or []
            if isinstance(feed, list):
                for row in feed[:10]:
                    if not isinstance(row, dict):
                        continue
                    published = row.get("time_published")
                    if not isinstance(published, str) or len(published) < 8:
                        continue
                    try:
                        day = datetime.strptime(published[:8], "%Y%m%d").date()
                        if day > cutoff_date:
                            report.fail(f"Future news date in trace[{idx}] NEWS_SENTIMENT: {day}")
                            break
                    except Exception:
                        continue


def check_trace_tool_policy(
    result: dict,
    *,
    forbid_price_fetch: bool,
    allow_news_fetch: bool,
    report: SanityReport,
) -> None:
    trace = result.get("trace") or []
    if not isinstance(trace, list):
        report.fail("trace must be a list")
        return

    for idx, event in enumerate(trace):
        if not isinstance(event, dict):
            continue
        tool = event.get("tool")
        function = str(event.get("function") or "").upper()
        if tool != "alpha_fetch":
            continue

        blocked = str(event.get("type") or "") == "policy_block"

        if forbid_price_fetch and function in FORBIDDEN_PRICE_FUNCTIONS:
            if blocked:
                report.warnings.append(f"Blocked price fetch attempt in trace[{idx}]: alpha_fetch/{function}")
            else:
                report.fail(f"Forbidden price fetch in trace[{idx}]: alpha_fetch/{function}")

        if function in ALLOWED_NEWS_FUNCTIONS and not allow_news_fetch:
            report.fail(f"News fetch disabled by policy but found in trace[{idx}]: alpha_fetch/{function}")

        if function not in ALLOWED_NEWS_FUNCTIONS and function not in FORBIDDEN_PRICE_FUNCTIONS:
            report.warnings.append(f"Unclassified alpha_fetch function in trace[{idx}]: {function}")


def run_sanity_checks(
    *,
    result: dict,
    cutoff_date: date,
    forbid_price_fetch: bool = True,
    allow_news_fetch: bool = True,
    enforce_runtime_cutoff: bool = False,
) -> SanityReport:
    report = SanityReport()
    check_required_fields(result, report)
    check_no_future_timestamps(result, cutoff_date, report, enforce_runtime_cutoff=enforce_runtime_cutoff)
    check_no_future_data_dates(result, cutoff_date, report)
    check_trace_tool_policy(
        result,
        forbid_price_fetch=forbid_price_fetch,
        allow_news_fetch=allow_news_fetch,
        report=report,
    )
    return report
