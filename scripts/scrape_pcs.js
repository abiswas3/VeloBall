const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');
const cheerio = require('cheerio');
const minimist = require('minimist');

function tableToCSV(tableHTML) {
  const $ = cheerio.load(tableHTML);
  const rows = [];
  $('tr').each((i, tr) => {
    const cols = [];
    $(tr).find('th, td').each((j, td) => {
      let text = $(td).text().trim();
      text = `"${text.replace(/"/g, '""')}"`;
      cols.push(text);
    });
    rows.push(cols.join(','));
  });
  return rows.join('\n');
}

async function scrapeStageTabs(race, year, stage) {
  const baseUrl = `https://www.procyclingstats.com/race/${race}/${year}/stage-${stage}`;
  console.log(`🔄 Loading ${baseUrl}`);

  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  page.setDefaultTimeout(30000);

  try {
    await page.goto(baseUrl, { waitUntil: 'networkidle2' });
    await page.waitForSelector('ul.tabs.tabnav');

    // Get tabs info
    const tabs = await page.evaluate(() =>
      Array.from(document.querySelectorAll('ul.tabs.tabnav li a.selectResultTab')).map(a => ({
        id: a.getAttribute('data-id'),
        text: a.textContent.trim().toLowerCase(),
      }))
    );

    console.log('Found tabs:', tabs.map(t => t.text).join(', '));

    const outDir = path.join('data', race.replace(/[- ]/g, '_'), String(year));
    fs.mkdirSync(outDir, { recursive: true });

    for (const tab of tabs) {
      // Click tab link
      await page.evaluate(id => {
        document.querySelector(`a.selectResultTab[data-id="${id}"]`).click();
      }, tab.id);

      // Wait for the tab's <li> to get class 'cur' (indicating it's active)
      await page.waitForFunction(
        id => {
          const li = document.querySelector(`a.selectResultTab[data-id="${id}"]`)?.parentElement;
          return li?.classList.contains('cur');
        },
        {},
        tab.id
      );

      // Now get the visible table inside the results container (only one table is visible at a time)
      const tableHTML = await page.evaluate(() => {
        const tables = Array.from(document.querySelectorAll('table.results'));
        const visibleTable = tables.find(t => t.offsetParent !== null);
        return visibleTable ? visibleTable.outerHTML : null;
      });

      if (!tableHTML) {
        console.warn(`⚠️ No visible results table found for tab ${tab.text}`);
        continue;
      }

      const csv = tableToCSV(tableHTML);
      const filename = `stage-${stage}${tab.text === 'stage' ? '' : '-' + tab.text}.csv`;
      const filepath = path.join(outDir, filename);
      fs.writeFileSync(filepath, csv);
      console.log(`✅ Saved ${filepath}`);
    }
  } catch (err) {
    console.error('❌ Error during scraping:', err);
  } finally {
    await browser.close();
  }
}

// Main function to parse flags and run all scrapes
async function main() {
  const args = minimist(process.argv.slice(2), {
    default: {
      race: 'tour-de-france',
      'start-year': 2023,
      'end-year': 2023,
      force: false,
      'start-stage': 1,
      'end-stage': 21,
    },
    boolean: ['force'],
  });

  const race = args.race;
  const startYear = parseInt(args['start-year'], 10);
  const endYear = parseInt(args['end-year'], 10);
  const force = args.force;
  const startStage = parseInt(args['start-stage'], 10);
  const endStage = parseInt(args['end-stage'], 10);

  for (let year = startYear; year <= endYear; year++) {
    for (let stage = startStage; stage <= endStage; stage++) {
      console.log(`\n===== Scraping ${race} ${year} stage ${stage} =====`);
      await scrapeStageTabs(race, year, stage);
    }
  }
}

main();
