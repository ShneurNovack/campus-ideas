"""Three calendars for every event date: Jewish (Hebrew date, holiday windows, parsha), secular (US holidays
that centers theme events around) and, per center, the academic term (inferred from the center's own activity
in prepare.py, because no API field gives term dates)."""
import datetime as dt
from functools import lru_cache
from pyluach import dates, parshios

HEB_MONTHS = {1: "Nisan", 2: "Iyar", 3: "Sivan", 4: "Tammuz", 5: "Av", 6: "Elul", 7: "Tishrei", 8: "Cheshvan",
              9: "Kislev", 10: "Tevet", 11: "Shvat", 12: "Adar", 13: "Adar"}

# key, label, (month, day) start, length in days, days before the start that already count as "the run-up"
JEWISH = [
    ("rosh_hashana", "Rosh Hashana", (6, 29), 3, 10),
    ("yom_kippur", "Yom Kippur", (7, 9), 2, 5),
    ("sukkot", "Sukkot", (7, 14), 8, 4),
    ("simchat_torah", "Simchat Torah", (7, 22), 2, 2),
    ("chanukah", "Chanukah", (9, 24), 9, 12),
    ("tu_bshvat", "Tu B'Shvat", (11, 14), 2, 5),
    ("purim", "Purim", ("adar", 13), 3, 14),
    ("pesach", "Pesach", (1, 14), 9, 14),
    ("yom_hashoah", "Yom HaShoah", (1, 26), 2, 3),
    ("yom_haatzmaut", "Yom Ha'atzmaut", (2, 3), 3, 4),
    ("lag_baomer", "Lag BaOmer", (2, 17), 2, 6),
    ("shavuot", "Shavuot", (3, 5), 3, 7),
    ("tisha_bav", "Tisha B'Av", (5, 8), 2, 3),
]
JEWISH_LABEL = {k: l for k, l, *_ in JEWISH}

SUPER_BOWL = {2020: (2, 2), 2021: (2, 7), 2022: (2, 13), 2023: (2, 12), 2024: (2, 11), 2025: (2, 9), 2026: (2, 8), 2027: (2, 14), 2028: (2, 13)}
SECULAR_LABEL = {"new_year": "New Year", "super_bowl": "Super Bowl", "valentines": "Valentine's Day", "st_patricks": "St. Patrick's Day",
                 "cinco_de_mayo": "Cinco de Mayo", "mothers_day": "Mother's Day", "july_4": "July 4", "sept_11": "9/11",
                 "oct_7": "October 7", "halloween": "Halloween", "thanksgiving": "Thanksgiving"}


def _nth_weekday(year, month, weekday, n):
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


@lru_cache(maxsize=None)
def jewish_windows(year):
    """Holiday windows that touch a civil year: list of (key, start, end, runup_start)."""
    out = []
    for hy in (year + 3759, year + 3760, year + 3761):
        leap = ((7 * hy) + 1) % 19 < 7
        for key, _label, (m, d), length, runup in JEWISH:
            if m == "adar":
                m = 13 if leap else 12
            try:
                start = dates.HebrewDate(hy, m, d).to_pydate()
            except Exception:
                start = dates.HebrewDate(hy, m, d - 1).to_pydate()
            if abs(start.year - year) > 1:
                continue
            out.append((key, start, start + dt.timedelta(days=length - 1), start - dt.timedelta(days=runup)))
    return sorted(set(out), key=lambda x: x[1])


@lru_cache(maxsize=None)
def secular_windows(year):
    w = []
    def add(key, d, before, after):
        w.append((key, d - dt.timedelta(days=before), d + dt.timedelta(days=after), d))
    add("new_year", dt.date(year, 1, 1), 1, 1)
    if year in SUPER_BOWL:
        add("super_bowl", dt.date(year, *SUPER_BOWL[year]), 3, 0)
    add("valentines", dt.date(year, 2, 14), 4, 1)
    add("st_patricks", dt.date(year, 3, 17), 3, 1)
    add("cinco_de_mayo", dt.date(year, 5, 5), 3, 1)
    add("mothers_day", _nth_weekday(year, 5, 6, 2), 3, 0)
    add("july_4", dt.date(year, 7, 4), 2, 1)
    add("sept_11", dt.date(year, 9, 11), 1, 1)
    if year >= 2024:
        add("oct_7", dt.date(year, 10, 7), 3, 3)
    add("halloween", dt.date(year, 10, 31), 5, 1)
    add("thanksgiving", _nth_weekday(year, 11, 3, 4), 8, 2)
    return w


@lru_cache(maxsize=None)
def context(d):
    """Everything the engine wants to know about one civil date."""
    if isinstance(d, dt.datetime):
        d = d.date()
    h = dates.HebrewDate.from_pydate(d)
    out = {"hy": h.year, "hm": h.month, "hd": h.day, "hmonth": HEB_MONTHS[h.month], "jh": None, "jh_runup": None, "jh_days": None, "sec": None}
    best = None
    for key, start, end, runup in jewish_windows(d.year):
        if start <= d <= end:
            out["jh"] = key
        elif runup <= d < start:
            days = (start - d).days
            if best is None or days < best[1]:
                best = (key, days)
    if best and not out["jh"]:
        out["jh_runup"], out["jh_days"] = best
    for key, a, b, _day in secular_windows(d.year):
        if a <= d <= b:
            out["sec"] = key
    # position in the Hebrew year as a day count from 1 Tishrei, for seasonal kernels on the Jewish calendar
    out["hdoy"] = (d - dates.HebrewDate(_hyear_start(h), 7, 1).to_pydate()).days
    out["hdoy_n"] = hdoy_norm(d)
    sat = d + dt.timedelta(days=(5 - d.weekday()) % 7)
    try:
        out["parsha"] = parshios.getparsha_string(dates.GregorianDate(sat.year, sat.month, sat.day), israel=False)
    except Exception:
        out["parsha"] = None
    return out


def _hyear_start(h):
    """pyluach counts the year from Nisan for months but the year number changes at Tishrei."""
    return h.year


def _leap(hy):
    return ((7 * hy) + 1) % 19 < 7


@lru_cache(maxsize=None)
def hdoy_norm(d):
    """Position in the Jewish year as days from 1 Tishrei, with a leap year's extra Adar folded away so that
    Purim, Pesach and everything after line up from year to year."""
    if isinstance(d, dt.datetime):
        d = d.date()
    h = dates.HebrewDate.from_pydate(d)
    n = (d - dates.HebrewDate(h.year, 7, 1).to_pydate()).days
    if _leap(h.year) and (h.month == 13 or h.month <= 6):
        n -= 30
    return n


def upcoming_jewish(d, horizon=45):
    """Holidays starting within the horizon: [(key, label, start, days)]."""
    out = []
    for y in (d.year, d.year + 1):
        for key, start, _end, _runup in jewish_windows(y):
            days = (start - d).days
            if 0 <= days <= horizon:
                out.append((key, JEWISH_LABEL[key], start, days))
    seen, res = set(), []
    for x in sorted(out, key=lambda x: x[3]):
        if x[0] not in seen:
            seen.add(x[0]); res.append(x)
    return res


def doy_gap(a, b, period=365):
    """Circular distance in days between two day-of-year positions."""
    g = abs(a - b) % period
    return min(g, period - g)
