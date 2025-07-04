import os
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import argparse



def get_stage_results(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="results")
    if not table:
        return None

    rows = table.find_all("tr")
    data = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 10:
            continue

        rider_cell = cols[7]
        rider_link = rider_cell.find("a")
        rider = rider_link.text.strip() if rider_link else rider_cell.text.strip()
        flag_span = rider_cell.find("span", class_="flag")
        nationality = flag_span["class"][1].upper() if flag_span and len(flag_span["class"]) > 1 else ""
        team = cols[8].text.strip()
        uci_points = cols[9].text.strip()

        row_data = {
            "rank": cols[0].text.strip(),
            "rider": rider,
            "nationality": nationality,
            "team": team,
            "uci_points": uci_points
        }

        if len(cols) > 12:
            # Stage results
            row_data["bonus"] = cols[11].text.strip()
            row_data["time"] = cols[12].text.strip()
        else:
            # GC-like tables
            row_data["prev"] = cols[1].text.strip()
            row_data["time"] = cols[11].text.strip()

        data.append(row_data)

    return pd.DataFrame(data)


def download_stage_variants(base_url, year, stage, out_dir, force=False):
    suffixes = {
        "": "stage",
        "-gc": "gc",
        "-points": "points",
        "-kom": "kom"
    }

    stage_str = str(stage)

    for suffix, label in suffixes.items():
        url = f"{base_url}{year}/stage-{stage_str}{suffix}"

        file_name = f"stage-{stage_str}.csv" if label == "stage" else f"stage-{stage_str}-{label}.csv"
        file_path = os.path.join(out_dir, file_name)

        if os.path.exists(file_path) and not force:
            print(f"⏭️  Skipping existing file for [{label}]: {file_path}")
            continue

        print(f"🔄 Scraping {label} from {url}")
        try:
            df = get_stage_results(url)
            if df is not None:
                df.to_csv(file_path, index=False)
                print(f"✅ Saved: {file_path}")
            else:
                print(f"⚠️  No results table at {url}")
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")

        time.sleep(1.5)
    print()


def main():
    parser = argparse.ArgumentParser(description="Scrape PCS stage results including GC, Points, KOM")
    parser.add_argument("race", nargs='?', default="tour-de-france",
                        help="Race name as in PCS URL (default: tour-de-france)")
    parser.add_argument("--start-year", type=int, default=2020,
                        help="Start year (default: 2020)")
    parser.add_argument("--end-year", type=int, default=2024,
                        help="End year (default: 2024)")
    parser.add_argument("--force", action="store_true",
                        help="Force scraping even if files already exist")
    args = parser.parse_args()

    race_for_url = args.race.lower().replace(" ", "-")
    race_for_path = args.race.replace("-", "_").replace(" ", "_")

    base_url = f"https://www.procyclingstats.com/race/{race_for_url}/"

    for year in range(args.start_year, args.end_year + 1):
        out_dir = os.path.join("data", race_for_path, str(year))
        os.makedirs(out_dir, exist_ok=True)

        for stage in range(1, 22):  # Tour usually has 21 stages
            download_stage_variants(base_url, year, stage, out_dir, force=args.force)



if __name__ == "__main__":
    main()

