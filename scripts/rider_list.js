const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

(async () => {
  const url = process.argv[2] || 'https://www.velogames.com/velogame/2024/riders.php';
  
  // Extract the year from the URL for output file naming, fallback to 'unknown' if no match
  const yearMatch = url.match(/velogame\/(\d{4})\//);
  const year = yearMatch ? yearMatch[1] : 'unknown';

  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  await page.goto(url, { waitUntil: 'networkidle2' });

  // Extract rows
  const rows = await page.$$eval('tr.even, tr.odd', trs => {
    return trs.map(tr => {
      const tds = tr.querySelectorAll('td');
      return {
        name: tds[1]?.innerText.trim(),
        team: tds[2]?.innerText.trim(),
        role: tds[3]?.innerText.trim(),
        races: tds[4]?.innerText.trim(),
        winPct: tds[5]?.innerText.trim(),
        points: tds[6]?.innerText.trim()
      };
    });
  });

  // Format CSV string
  const csv = [
    ['Name', 'Team', 'Role', 'Cost', 'Win %', 'Points'].join(','),
    ...rows.map(r => [r.name, r.team, r.role, r.races, r.winPct, r.points].join(','))
  ].join('\n');

  const outputDir = path.join('velostats', 'tour_de_france', year);
  const outputPath = path.join(outputDir, 'riders.csv');

  // Create directory if it doesn't exist
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  fs.writeFileSync(outputPath, csv);

  console.log(`Data saved to ${outputPath}`);
  await browser.close();
})();

