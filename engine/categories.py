"""Center groups.

Centers are grouped by what they do, never by what kind of school they sit on:

  size       how many sign up at its biggest events          the five biggest events of three years, averaged
  typical    how many sign up at an ordinary event           the median event that took sign-ups, last two years
  pace       how many events it holds a week                 events in the last year over the weeks it was in session
  jewish     how many Jewish students it serves              the counts each center entered, over all its campuses
  share      how much of that population it draws            biggest draw over Jewish students served
  strength   which kinds of event land high among its own    each kind's place among the center's events,
                                                             compared with where that kind lands across the network
  signups    how it uses the sign-up system
  stage      where it is in its life on the system

Each group is a band on one measure, with the cut points written down. The engine also tests whether centers
fall into natural clumps when all the measures are taken together (`structure` in centers.py). They do not, beyond
three size bands, so no blended clusters are published. "Centers like this one" is answered by nearest neighbors
instead (neighbors.py), which needs no borders.
"""
import numpy as np

MEALS = ["shabbat_dinner", "shabbat_lunch", "holiday_meal"]
HOLIDAYS = ["holiday_event", "holiday_service"]
PROGRAMS = ["class", "meal", "social", "cooking"]
STRENGTH_EDGE = 0.04       # a family must land this many percentile points (0.04 = 4) ahead of the center's other families
MIN_SCORED = 40            # scored events needed before a center's strengths are read

FACETS = [
    {"key": "size", "label": "Biggest events", "question": "How many sign up when it goes big?",
     "basis": "The average sign-ups of the center's five biggest events in the last three years. Big events are where nearly every center takes sign-ups, so this is the fairest single measure of how many students a center can bring out.",
     "values": [("r1", "Under 25", "Its five biggest events averaged fewer than 25 sign-ups."), ("r2", "25 to 59", "Its five biggest events averaged 25 to 59 sign-ups."),
                ("r3", "60 to 149", "Its five biggest events averaged 60 to 149 sign-ups."), ("r4", "150 or more", "Its five biggest events averaged 150 sign-ups or more."),
                ("unknown", "Sign-ups too rare to tell", "Fewer than 15 events with sign-ups in three years, or under 15% of events took any.")]},
    {"key": "typical", "label": "Typical event", "question": "How many sign up for an ordinary event?",
     "basis": "The median sign-ups of the center's events in the last two years, counting only events that took sign-ups. Centers that only take sign-ups for big events will read high here, which is why the biggest-events measure sits beside it.",
     "values": [("t1", "Under 5", "A typical event with sign-ups draws fewer than 5."), ("t2", "5 to 9", "A typical event draws 5 to 9."),
                ("t3", "10 to 19", "A typical event draws 10 to 19."), ("t4", "20 or more", "A typical event draws 20 or more."),
                ("unknown", "Sign-ups too rare to tell", "Fewer than 15 events with sign-ups in three years, or under 15% of events took any.")]},
    {"key": "pace", "label": "Events a week", "question": "How many events does it hold a week?",
     "basis": "Events in the last twelve months divided by the number of weeks in which the center held anything, so breaks do not drag the number down. Standing forms and to-go offers are not counted.",
     "values": [("p1", "About one a week", "Fewer than 1.5 events a week while in session."), ("p2", "Two a week", "1.5 to 2.4 events a week."),
                ("p3", "Three a week", "2.5 to 3.4 events a week."), ("p4", "Four or more a week", "3.5 events a week or more."),
                ("unknown", "Too few events to tell", "Fewer than 8 events in the last twelve months.")]},
    {"key": "jewish", "label": "Jewish students", "question": "How many Jewish students does it serve?",
     "basis": "The Jewish undergraduate and graduate counts each center entered in the Chabad on Campus directory, added up over every campus the center serves. These are the centers' own estimates.",
     "values": [("j1", "Under 300", "Fewer than 300 Jewish students."), ("j2", "300 to 999", "300 to 999 Jewish students."), ("j3", "1,000 to 2,499", "1,000 to 2,499 Jewish students."),
                ("j4", "2,500 or more", "2,500 Jewish students or more."), ("unknown", "Not listed", "No usable count in the directory.")]},
    {"key": "share", "label": "Share it draws", "question": "How much of its Jewish population comes to its biggest events?",
     "basis": "Biggest-event sign-ups divided by Jewish students served. It puts a center of 300 Jewish students and a center of 5,000 on the same footing.",
     "values": [("d1", "Under 3%", "Its biggest events draw fewer than 3 in 100 of the Jewish students it serves."), ("d2", "3% to 7%", "Its biggest events draw 3 to 7 in 100."),
                ("d3", "8% to 19%", "Its biggest events draw 8 to 19 in 100."), ("d4", "20% or more", "Its biggest events draw 1 in 5 or more."),
                ("unknown", "Cannot be worked out", "No Jewish student count, or sign-ups too rare.")]},
    {"key": "strength", "label": "What does well here", "question": "Which kinds of event run above this center's own usual?",
     "basis": "Every event is placed among all of its own center's events, from smallest to biggest. Each kind of event is then compared with where that kind lands at the average center. A center gets a strength when one family of events lands clearly higher, against the network, than its other families do.",
     "values": [("meals", "Shabbat and holiday meals", "Its Shabbat dinners, Shabbat lunches and holiday meals land higher among its own events than they do at most centers."),
                ("holidays", "Holiday events and services", "Its holiday events and services land higher among its own events than they do at most centers."),
                ("programs", "Classes, socials and food events", "Its classes, socials, food hangouts and baking land higher among its own events than they do at most centers."),
                ("even", "No single strength", "No family of events stands out from the others."),
                ("unknown", "Too few scored events to tell", "Fewer than 40 scored events in three years.")]},
    {"key": "signups", "label": "Use of sign-ups", "question": "How does it use the sign-up system?",
     "basis": "The share of the last three years' events that took at least one sign-up. This decides how much of the center's calendar can be read at all.",
     "values": [("s4", "Nearly everything", "80% or more of its events took sign-ups."), ("s3", "Most events", "50% to 79% took sign-ups."), ("s2", "Some events", "15% to 49% took sign-ups."),
                ("s1", "Rarely", "Under 15% took sign-ups. The system is used as a calendar."), ("unknown", "Too few events to tell", "Fewer than 15 events in the last three years.")]},
    {"key": "stage", "label": "Stage on the system", "question": "Where is it in its life on the sign-up system?",
     "basis": "First and latest event, and the last twelve months of events against the twelve before.",
     "values": [("new", "New to the system", "Its first event on the sign-up system was in the last 18 months. The center itself may be much older."),
                ("growing", "Growing", "At least 30% more events in the last twelve months than the twelve before."), ("steady", "Steady", "About as many events as the year before."),
                ("slowing", "Slowing", "At least 30% fewer events than the year before."), ("quiet", "Quiet on the system", "Nothing held or posted on the sign-up system in the last six months.")]},
]
LABEL = {f["key"]: {k: lab for k, lab, _ in f["values"]} for f in FACETS}
ORDER = {f["key"]: [k for k, _, _ in f["values"]] for f in FACETS}


def _known(row):
    return row["measured3"] >= 15 and row["share3"] >= 0.15


def size(row):
    v = row["top5"]
    if v != v or not _known(row):
        return "unknown"
    return "r1" if v < 25 else "r2" if v < 60 else "r3" if v < 150 else "r4"


def typical(row):
    v = row["usual"]
    if v != v or not _known(row):
        return "unknown"
    return "t1" if v < 5 else "t2" if v < 10 else "t3" if v < 20 else "t4"


def pace(row):
    v = row["pace"]
    if v != v:
        return "unknown"
    return "p1" if v < 1.5 else "p2" if v < 2.5 else "p3" if v < 3.5 else "p4"


def jewish(row):
    n = row["jpop"]
    if n != n or n <= 0:
        return "unknown"
    return "j1" if n < 300 else "j2" if n < 1000 else "j3" if n < 2500 else "j4"


def share(row):
    v = row["draw_share"]
    if v != v or not _known(row):
        return "unknown"
    return "d1" if v < 0.03 else "d2" if v < 0.08 else "d3" if v < 0.20 else "d4"


def family_strengths(row):
    """How far each family of events runs ahead of the center's other families (log), or None when unread."""
    out = {}
    for key, kinds in (("meals", MEALS), ("holidays", HOLIDAYS), ("programs", PROGRAMS)):
        w = np.array([min(row["nfam"][k], 10) for k in kinds], float)
        out[key] = float((np.array([row["strength"][k] for k in kinds]) * w).sum() / w.sum()) if w.sum() >= 5 else None
    have = [v for v in out.values() if v is not None]
    if len(have) < 2:
        return None
    mean = float(np.mean(have))
    return {k: (v - mean if v is not None else None) for k, v in out.items()}


def strength(row):
    if row["scored3"] < MIN_SCORED:
        return "unknown"
    fs = family_strengths(row)
    if not fs:
        return "unknown"
    best = max((k for k in fs if fs[k] is not None), key=lambda k: fs[k])
    return best if fs[best] >= STRENGTH_EDGE else "even"


def signups(row):
    if row["n3"] < 15:
        return "unknown"
    s = row["share3"]
    return "s4" if s >= 0.8 else "s3" if s >= 0.5 else "s2" if s >= 0.15 else "s1"


def stage(first, last, n_now, n_before, today):
    if first is None or last is None or (today - last).days > 183:
        return "quiet"
    if (today - first).days <= 548:
        return "new"
    if n_before >= 20:
        ratio = n_now / n_before
        return "growing" if ratio >= 1.3 else "slowing" if ratio <= 0.7 else "steady"
    return "growing" if n_now >= 40 else "steady"
