# ============================================================
#  pro-football-ai / scraper/soccerstats_scraper.py
#  SoccerSTATS.com scraper — fetches team stats server-side
# ============================================================

import re
import time
import json
import urllib.request
import urllib.parse
from typing import Optional, List

from config import (SOCCERSTATS_BASE, SCRAPER_DELAY, SCRAPER_TIMEOUT,
                    SCRAPER_USER_AGENT, LEAGUES)


def _fetch(url: str, retries: int = 3) -> str:
    """Fetch a URL with retry logic."""
    headers = {
        "User-Agent":      SCRAPER_USER_AGENT,
        "Accept":          "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity",
        "Connection":      "keep-alive",
        "Cache-Control":   "no-cache",
    }
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=SCRAPER_TIMEOUT) as resp:
                charset = resp.headers.get_content_charset("utf-8") or "utf-8"
                return resp.read().decode(charset, errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise Exception(f"SoccerSTATS blocked the request (403). Try again later.")
            if e.code == 404:
                raise Exception(f"Page not found (404): {url}")
            if attempt == retries - 1:
                raise Exception(f"HTTP {e.code} fetching {url}")
            time.sleep(SCRAPER_DELAY * 2)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(SCRAPER_DELAY)

    raise Exception(f"Failed to fetch {url} after {retries} attempts")


def _strip_tags(html: str) -> str:
    """Strip HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", "", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text.strip()


def _parse_float(s: str) -> Optional[float]:
    try:
        return float(s.replace("%", "").replace(",", ".").strip())
    except (ValueError, AttributeError):
        return None


# ── SEARCH TEAMS ─────────────────────────────────────────

def search_teams(league: str, query: str = "") -> List[dict]:
    """
    Search for teams in a league on SoccerSTATS.

    Returns:
        list of {"name": ..., "stats": ..., "league": ...}
    """
    url = f"{SOCCERSTATS_BASE}/latest.asp?league={league}&pmtype=byteam"
    try:
        html = _fetch(url)
        time.sleep(SCRAPER_DELAY)
    except Exception as e:
        return [{"error": str(e)}]

    teams = []
    seen  = set()

    # Pattern: teamstats.asp?league=england&stats=u324-arsenal
    pattern = re.compile(
        r'teamstats\.asp\?league=([^&"]+)&(?:amp;)?stats=([^"&\s]+)"[^>]*>\s*([^<]{2,50})\s*</a>',
        re.IGNORECASE
    )

    for m in pattern.finditer(html):
        lg, stats_id, name = m.group(1), m.group(2), _strip_tags(m.group(3))
        key = stats_id.lower()
        if key in seen:
            continue
        seen.add(key)
        if not query or query.lower() in name.lower():
            teams.append({"name": name, "stats": stats_id, "league": lg})

    return teams[:15]


# ── PARSE STATS TABLE ────────────────────────────────────

def _parse_stats_table(html: str) -> dict:
    """
    Parse the main statistics table from a SoccerSTATS team page.
    Returns dict with keys: overall, home, away, fh, sh
    """
    sections = {}

    # Find all table rows
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL)

    for row in rows:
        # Extract all cell contents
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.IGNORECASE | re.DOTALL)
        if len(cells) < 4:
            continue

        # Clean cells
        clean = [_strip_tags(c) for c in cells]
        if not clean:
            continue

        label = clean[0].lower().strip()

        # Identify row type
        if "overall" in label or ("all" in label and "result" not in label):
            key = "overall"
        elif label == "home" or "home game" in label:
            key = "home"
        elif label == "away" or "away game" in label:
            key = "away"
        elif "1st" in label or "first half" in label or label.startswith("ht"):
            key = "fh"
        elif "2nd" in label or "second half" in label:
            key = "sh"
        else:
            continue

        # Parse numeric values
        nums = []
        pcts = []
        for i, c in enumerate(clean[1:], 1):
            f = _parse_float(c)
            if f is not None:
                if "%" in c:
                    pcts.append((i, f))
                else:
                    nums.append((i, f))

        if not nums:
            continue

        parsed = _extract_section_stats(nums, pcts, clean[1:])
        sections[key] = parsed

    return sections


def _extract_section_stats(nums: list, pcts: list, raw_cells: list) -> dict:
    """Extract GP, GF, GA, TG, FTS, CS, BTS, 1.5+, 2.5+, 3.5+ from parsed cells."""
    result = {}

    # GP = first integer > 1
    for idx, val in nums:
        if val == int(val) and 1 < val < 200:
            result["gp"] = int(val)
            break

    # Find GF+GA+TG triplet (TG ≈ GF+GA)
    small = [(i, v) for i, v in nums if 0.1 < v < 8.0]

    for j in range(len(small) - 2):
        a_i, a = small[j]
        b_i, b = small[j+1]
        c_i, c = small[j+2]
        if abs((a + b) - c) < 0.15:
            result["gf"] = round(a, 3)
            result["ga"] = round(b, 3)
            result["tg"] = round(c, 3)
            break

    if "gf" not in result and len(small) >= 2:
        result["gf"] = round(small[0][1], 3)
        result["ga"] = round(small[1][1], 3)
        result["tg"] = round(result["gf"] + result["ga"], 3)

    # Percentages — typical SoccerSTATS column order:
    # W% | D% | L% | FTS% | CS% | BTS% | 1.5+ | 2.5+ | 3.5+
    if pcts:
        pct_vals = [v for _, v in sorted(pcts, key=lambda x: x[0])]
        # Filter out win/draw/loss percentages (usually 0-100 and sum to ~100)
        filtered = [v for v in pct_vals if 0 <= v <= 100]
        if len(filtered) >= 4:
            # Skip first 3 (W/D/L) if they sum close to 100
            if len(filtered) >= 6:
                first3 = filtered[:3]
                if 85 <= sum(first3) <= 115:
                    filtered = filtered[3:]

            if len(filtered) >= 1: result["fts"]  = filtered[0]
            if len(filtered) >= 2: result["cs"]   = filtered[1]
            if len(filtered) >= 3: result["bts"]  = filtered[2]
            if len(filtered) >= 4: result["o15"]  = filtered[3]
            if len(filtered) >= 5: result["o25"]  = filtered[4]
            if len(filtered) >= 6: result["o35"]  = filtered[5]

    return result


# ── MAIN TEAM STATS FETCH ────────────────────────────────

def get_team_stats(league: str, stats_id: str) -> dict:
    """
    Fetch full team statistics from SoccerSTATS.

    Returns:
        dict with all stats fields used by the prediction model
    """
    url = f"{SOCCERSTATS_BASE}/teamstats.asp?league={league}&stats={stats_id}"

    try:
        html = _fetch(url)
        time.sleep(SCRAPER_DELAY)
    except Exception as e:
        raise Exception(f"Failed to fetch team stats: {e}")

    # Extract team name
    name_m = re.search(r"<h1[^>]*>([^<]+)", html, re.IGNORECASE)
    team_name = name_m.group(1).strip() if name_m else stats_id
    # Clean up name
    team_name = re.sub(r"\s*(statistics|stats|football).*", "", team_name, flags=re.IGNORECASE).strip()

    # Parse stats table
    sections = _parse_stats_table(html)

    # Fallback: try data-stat attributes
    if not sections:
        sections = _parse_data_stats(html)

    if not sections.get("overall") and not sections.get("home"):
        # Last resort: grep for GF/GA patterns
        gf_m = re.search(r"(?:goals?.*?for|GF)[^0-9]*(\d+\.\d+)", html, re.IGNORECASE)
        ga_m = re.search(r"(?:goals?.*?against|GA)[^0-9]*(\d+\.\d+)", html, re.IGNORECASE)
        if gf_m and ga_m:
            sections["overall"] = {
                "gf": float(gf_m.group(1)),
                "ga": float(ga_m.group(1)),
                "tg": float(gf_m.group(1)) + float(ga_m.group(1))
            }
        else:
            raise Exception(
                f"Could not parse stats for '{team_name}'. "
                "SoccerSTATS may have changed their page structure."
            )

    ov = sections.get("overall", {})
    hm = sections.get("home",    {})
    aw = sections.get("away",    {})
    fh = sections.get("fh",      {})

    # Build the final stats dict
    gf = ov.get("gf", 1.3)
    ga = ov.get("ga", 1.3)

    stats = {
        "name":   team_name,
        "league": league,
        "source": "soccerstats",
        "url":    url,

        # Overall
        "gp":  ov.get("gp",  20),
        "gf":  gf,
        "ga":  ga,
        "tg":  ov.get("tg",  gf + ga),
        "fts": ov.get("fts", 22),
        "cs":  ov.get("cs",  25),
        "bts": ov.get("bts", 45),
        "o15": ov.get("o15", 72),
        "o25": ov.get("o25", 48),
        "o35": ov.get("o35", 26),

        # Home/Away split
        "hgf": hm.get("gf", gf * 1.15),
        "hga": hm.get("ga", ga * 0.90),
        "agf": aw.get("gf", gf * 0.85),
        "aga": aw.get("ga", ga * 1.10),

        # First Half
        "fhgf":  fh.get("gf",  round(gf * 0.44, 3)),
        "fhga":  fh.get("ga",  round(ga * 0.44, 3)),
        "fhtg":  fh.get("tg",  round((gf + ga) * 0.44, 3)),
        "fhbts": fh.get("bts", 20),
    }

    return stats


def _parse_data_stats(html: str) -> dict:
    """Alternative parser using data-stat attributes."""
    sections = {}
    gf_vals, ga_vals = [], []

    for m in re.finditer(r'data-stat="goals_for"[^>]*>(\d+\.?\d*)', html):
        gf_vals.append(float(m.group(1)))
    for m in re.finditer(r'data-stat="goals_against"[^>]*>(\d+\.?\d*)', html):
        ga_vals.append(float(m.group(1)))

    if gf_vals and ga_vals:
        sections["overall"] = {"gf": gf_vals[0], "ga": ga_vals[0], "tg": gf_vals[0] + ga_vals[0]}
    return sections


# ── LEAGUE TABLE ─────────────────────────────────────────

def get_league_stats(league: str) -> List[dict]:
    """Fetch all teams and their stats for a full league."""
    teams = search_teams(league)
    results = []

    for t in teams:
        try:
            stats = get_team_stats(league, t["stats"])
            results.append(stats)
            time.sleep(SCRAPER_DELAY)
        except Exception as e:
            print(f"⚠️  Could not fetch {t['name']}: {e}")

    return results


if __name__ == "__main__":
    # Quick test
    import sys
    league = sys.argv[1] if len(sys.argv) > 1 else "england"
    query  = sys.argv[2] if len(sys.argv) > 2 else ""

    print(f"🔍 Searching teams in '{league}' for '{query}'...")
    teams = search_teams(league, query)
    for t in teams:
        print(f"  → {t['name']} [{t['stats']}]")
