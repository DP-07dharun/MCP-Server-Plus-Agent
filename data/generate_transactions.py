"""Generate a synthetic financial-transactions CSV with injected fraud patterns.

    python data/generate_transactions.py [--rows 5000] [--seed 42] [--anomalies 40]

Adapted from the generator of the spark-financial-transactions-analysis project, with
labelled anomalies added so the MCP server's fraud checks can be evaluated.

The data is random but *structured* so the analyses have something to find:
  * every vendor belongs to one category, and amounts follow a per-category distribution;
  * each user has a home city (most transactions happen there) and 1-2 bank accounts,
    each account held at one financial institution;
  * spending is seasonal (more in December and summer, travel peaks in July/August).
  * --anomalies N injects N fraud scenarios (amount spikes, card-testing bursts,
    night-time transactions away from home, impossible travel). Their transaction IDs
    and scenario names are written to anomalies_truth.csv, which is NOT given to the agent.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

HEADER = [
    "user_id",
    "transaction_id",
    "bank_account_id",
    "financial_institution_id",
    "date",
    "third_party",
    "category",
    "transaction_amount",
    "city",
    "payment_method",
    "currency",
]

# category -> (vendors, median amount EUR, spread)
CATEGORIES: dict[str, tuple[list[str], float, float]] = {
    "Food": (["Carrefour", "Leclerc", "Auchan", "Lidl", "Monoprix", "Picard"], 38.0, 0.6),
    "Restaurants": (["McDonalds", "Starbucks", "Deliveroo", "UberEats", "Paul"], 22.0, 0.6),
    "Travel": (["SNCF", "Air France", "Booking.com", "Airbnb", "BlaBlaCar"], 180.0, 0.8),
    "Shopping": (["Amazon", "Fnac", "Zara", "Decathlon", "IKEA", "H&M"], 65.0, 0.8),
    "Health": (["Pharmacie Centrale", "Doctolib", "Optic 2000", "Clinique Pasteur"], 45.0, 0.7),
    "Entertainment": (["Netflix", "Spotify", "UGC Cinemas", "Steam", "Ticketmaster"], 18.0, 0.7),
    "Utilities": (["EDF", "Engie", "Orange", "SFR", "Veolia"], 75.0, 0.4),
    "Transport": (["TotalEnergies", "RATP", "Uber", "Lime", "Shell"], 32.0, 0.6),
    "Education": (["Udemy", "Coursera", "Cultura", "Gibert Joseph"], 55.0, 0.7),
}

# Relative weight of each category in the number of transactions.
CATEGORY_WEIGHTS = {
    "Food": 24,
    "Restaurants": 16,
    "Shopping": 14,
    "Transport": 13,
    "Entertainment": 9,
    "Utilities": 8,
    "Health": 7,
    "Travel": 5,
    "Education": 4,
}

CITIES = {  # city -> population weight
    "Paris": 30,
    "Lyon": 12,
    "Marseille": 11,
    "Toulouse": 9,
    "Nice": 7,
    "Nantes": 7,
    "Lille": 7,
    "Bordeaux": 7,
    "Strasbourg": 5,
    "Rouen": 5,
}

INSTITUTIONS = {  # financial_institution_id -> market-share weight
    101: 22,  # e.g. BNP Paribas
    102: 18,  # Credit Agricole
    103: 15,  # Societe Generale
    104: 12,  # LCL
    105: 10,  # Caisse d'Epargne
    106: 8,  # Banque Populaire
    107: 6,  # La Banque Postale
    108: 5,  # Boursorama
    109: 4,  # Revolut
}

# Approximate (lat, lon) of each city, used for the impossible-travel scenario.
CITY_COORDS = {
    "Paris": (48.86, 2.35),
    "Lyon": (45.76, 4.84),
    "Marseille": (43.30, 5.37),
    "Toulouse": (43.60, 1.44),
    "Nice": (43.70, 7.27),
    "Nantes": (47.22, -1.55),
    "Lille": (50.63, 3.06),
    "Bordeaux": (44.84, -0.58),
    "Strasbourg": (48.57, 7.75),
    "Rouen": (49.44, 1.10),
}

SCENARIOS = ("amount_spike", "card_testing", "night_away_from_home", "impossible_travel")

# Seasonality multiplier for the number of transactions per month (Jan..Dec).
MONTH_WEIGHTS = [0.85, 0.8, 0.9, 0.95, 1.0, 1.05, 1.2, 1.15, 0.95, 0.95, 1.05, 1.4]
PAYMENT_METHODS = {"Card": 62, "Contactless": 18, "Transfer": 10, "Direct Debit": 7, "Mobile": 3}


def weighted(rng: random.Random, table: dict) -> object:
    keys = list(table)
    return rng.choices(keys, weights=[table[k] for k in keys], k=1)[0]


def generate(
    rows: int, seed: int, year: int = 2025, users: int = 250, anomalies: int = 0
) -> tuple[list[list[str]], list[tuple[str, str]]]:
    """Return (records, truth) where truth lists (transaction_id, scenario) of injected fraud."""
    rng = random.Random(seed)

    # Users: a home city and 1-2 accounts, each at one institution.
    profiles = []
    for u in range(1, users + 1):
        accounts = []
        for _ in range(1 if rng.random() < 0.7 else 2):
            accounts.append((f"ACC{rng.randint(10_000_000, 99_999_999)}", weighted(rng, INSTITUTIONS)))
        activity = rng.lognormvariate(0, 0.6)  # some users transact far more than others
        profiles.append((f"U{u:04d}", weighted(rng, CITIES), accounts, activity))
    user_weights = [p[3] for p in profiles]

    start = datetime(year, 1, 1)
    days_in_year = (datetime(year + 1, 1, 1) - start).days
    day_weights = [MONTH_WEIGHTS[(start + timedelta(days=d)).month - 1] for d in range(days_in_year)]

    records = []
    for _ in range(rows):
        user_id, home, accounts, _ = rng.choices(profiles, weights=user_weights, k=1)[0]
        account, institution = rng.choice(accounts)
        category = weighted(rng, CATEGORY_WEIGHTS)
        day = rng.choices(range(days_in_year), weights=day_weights, k=1)[0]
        when = start + timedelta(
            days=day, hours=rng.randint(7, 22), minutes=rng.randint(0, 59), seconds=rng.randint(0, 59)
        )
        if category == "Travel" and when.month not in (7, 8, 12) and rng.random() < 0.5:
            category = "Shopping"  # thin out travel outside holiday months
        vendors, median, spread = CATEGORIES[category]
        amount = rng.lognormvariate(math.log(median), spread)
        if when.month == 12 and category in ("Shopping", "Entertainment"):
            amount *= 1.35  # holiday spending
        city = home if rng.random() < 0.8 else weighted(rng, CITIES)
        records.append(
            [
                user_id,
                "",  # transaction_id, assigned after sorting
                account,
                str(institution),
                when.strftime("%Y-%m-%d %H:%M:%S"),
                rng.choice(vendors),
                category,
                f"{max(amount, 1.0):.2f}",
                city,
                weighted(rng, PAYMENT_METHODS),
                "EUR",
                "",  # hidden label column, removed before writing
            ]
        )
    for n in range(anomalies):
        records.extend(_inject(rng, profiles, start, days_in_year, SCENARIOS[n % len(SCENARIOS)], n))
    records.sort(key=lambda r: r[4])  # chronological, like a bank export
    truth = []
    for i, record in enumerate(records, start=1):
        record[1] = f"TXN{i:07d}"  # IDs follow time order
        label = record.pop()
        if label:
            truth.append((record[1], label))
    return records, truth


def _row(user, account, institution, when, vendor, category, amount, city, method, label):
    return [user, "", account, str(institution), when.strftime("%Y-%m-%d %H:%M:%S"),
            vendor, category, f"{amount:.2f}", city, method, "EUR", label]


def _inject(rng, profiles, start, days_in_year, scenario, n):
    """Build the rows of one fraud scenario for a random user."""
    user, home, accounts, _ = rng.choice(profiles)
    account, institution = rng.choice(accounts)
    day = start + timedelta(days=rng.randrange(days_in_year - 1))
    label = f"{scenario}#{n}"
    if scenario == "amount_spike":  # one purchase far above anything normal for the category
        category = rng.choice(["Shopping", "Travel", "Entertainment"])
        vendors, median, _ = CATEGORIES[category]
        when = day + timedelta(hours=rng.randint(9, 20), minutes=rng.randint(0, 59))
        amount = median * rng.uniform(25, 60)
        return [_row(user, account, institution, when, rng.choice(vendors), category, amount,
                     home, "Card", label)]
    if scenario == "card_testing":  # a burst of tiny online charges to probe a stolen card
        when = day + timedelta(hours=rng.randint(0, 23), minutes=rng.randint(0, 40))
        vendor = rng.choice(["Steam", "Amazon", "Spotify"])
        category = "Entertainment" if vendor != "Amazon" else "Shopping"
        rows = []
        for _ in range(rng.randint(5, 8)):
            when += timedelta(seconds=rng.randint(20, 90))
            rows.append(_row(user, account, institution, when, vendor, category,
                             rng.uniform(0.5, 1.99), home, "Mobile", label))
        return rows
    if scenario == "night_away_from_home":  # large night-time spend in another city
        city = rng.choice([c for c in CITIES if c != home])
        when = day + timedelta(hours=rng.randint(1, 4), minutes=rng.randint(0, 59))
        vendors, median, _ = CATEGORIES["Shopping"]
        return [_row(user, account, institution, when, rng.choice(vendors), "Shopping",
                     median * rng.uniform(6, 15), city, "Contactless", label)]
    # impossible_travel: two card payments in distant cities a few minutes apart
    far = [c for c in CITIES if _km(home, c) > 450]
    city = rng.choice(far)
    when = day + timedelta(hours=rng.randint(9, 19), minutes=rng.randint(0, 59))
    vendors, median, _ = CATEGORIES["Food"]
    first = _row(user, account, institution, when, rng.choice(vendors), "Food",
                 median * rng.uniform(0.8, 1.5), home, "Card", label)
    when += timedelta(minutes=rng.randint(5, 25))
    second = _row(user, account, institution, when, rng.choice(vendors), "Food",
                  median * rng.uniform(1, 3), city, "Card", label)
    return [first, second]


def _km(a: str, b: str) -> float:
    (la1, lo1), (la2, lo2) = CITY_COORDS[a], CITY_COORDS[b]
    la1, lo1, la2, lo2 = map(math.radians, (la1, lo1, la2, lo2))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anomalies", type=int, default=40, help="fraud scenarios to inject")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("transactions.csv"))
    args = parser.parse_args()
    if args.rows < 1000:
        parser.error("the project requires at least 1000 records")

    rows, truth = generate(args.rows, args.seed, anomalies=args.anomalies)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    truth_path = args.out.with_name("anomalies_truth.csv")
    with truth_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["transaction_id", "scenario"])
        writer.writerows(truth)
    print(f"Wrote {len(rows)} transactions to {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    print(f"Wrote {len(truth)} labelled fraud transactions to {truth_path}")


if __name__ == "__main__":
    main()
