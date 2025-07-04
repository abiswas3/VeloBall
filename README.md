# velo_scraper

[![Python package](https://github.com/abiswas3/VeloBall/actions/workflows/python-package.yml/badge.svg)](https://github.com/abiswas3/VeloBall/actions/workflows/python-package.yml)
This project uses the pcs-scraper binary (part of the velo-scraper package) to politely scrape professional cycling statistics for the Tour de France, Vuelta a España, and Giro d’Italia. The data is used solely for Velogames analysis and community purposes. I do not profit from any of the data or this project. If ProCyclingStats has any issue with this scraper, I will take it down immediately. The code is open source and licensed under the MIT License, as detailed in the LICENSE.txt file.

-----

## Table of Contents

- [Installation](#installation)
- [License](LICENSE.txt)

## Installation

To install `velo_scraper`, clone the repository and install it in editable mode:

```console
git clone https://github.com/abiswas3/VeloBall.git
cd velo_scraper
pip install -e .
```

## Getting Historical data

To get tour data from 2022 to 2024 run the following lines to pull data from pro cycling stats.
```bash
pcs-scraper --race tour-de-france --start-year 2022 --end-year 2024
```
For more options
```bash
pcs-scraper --help
```

for more features.

## Getting Current Year Data

Run the scraper using the default URL hardcoded in the script (currently for 2025 riders):

```bash
npm install
npm run scrape-riders
```

To specify a specific URL run 

```bash
node scripts/rider_list.js https://www.velogames.com/velogame/2024/riders.php
```

To get PCS data
```bash
node scripts/scrape_pcs.js --force --start-year=2023 --end-year=2024 --race=tour-de-france --start-stage=1 --end-stage=21
```
