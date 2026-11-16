const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const screenshots = process.argv.includes('--screenshots');
const artifacts = path.join(__dirname, 'artifacts');
const types = { '.html': 'text/html', '.json': 'application/json', '.jpg': 'image/jpeg', '.png': 'image/png' };

async function main() {
  const server = http.createServer(async (req, res) => {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    const relative = decodeURIComponent(pathname === '/' ? 'index.html' : pathname.slice(1));
    const file = path.resolve(root, relative);
    if (!file.startsWith(root + path.sep) || !types[path.extname(file)]) {
      res.writeHead(404).end();
      return;
    }
    try {
      const content = await fs.readFile(file);
      res.writeHead(200, { 'Content-Type': types[path.extname(file)] }).end(content);
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    // Tests are deterministic and do not download NASA imagery or call live APIs.
    await page.route(/^https?:\/\//, (route) => {
      if (new URL(route.request().url()).hostname === '127.0.0.1') return route.continue();
      return route.abort();
    });
    if (screenshots) await fs.mkdir(artifacts, { recursive: true });
    const url = `http://127.0.0.1:${server.address().port}/`;
    await page.goto(url);
    await page.waitForSelector('#boot.done', { state: 'attached' });
    assert.equal(await page.locator('.station-card').count(), 10);
    assert.match(await page.locator('#layerProvenanceBadge').innerText(), /CALIBRATED TWIN/);

    for (const [width, height] of [[1440, 900], [1440, 768], [1280, 720], [1024, 900], [390, 900]]) {
      await page.setViewportSize({ width, height });
      await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const layout = await page.evaluate(() => {
        const box = (selector) => {
          const r = document.querySelector(selector).getBoundingClientRect();
          return { x: r.x, y: r.y, right: r.right, bottom: r.bottom, width: r.width, height: r.height };
        };
        return {
          overflow: document.documentElement.scrollWidth > innerWidth,
          header: box('.topbar'), center: box('.center-stage'),
          mission: box('.sidebar-left'), telemetry: box('.sidebar-right'),
          map: box('#map'), profile: box('.profile-strip'),
          navy: getComputedStyle(document.querySelector('.topbar')).backgroundColor,
        };
      });
      assert.equal(layout.overflow, false, `horizontal overflow at ${width}px`);
      assert.equal(layout.navy, 'rgb(7, 23, 47)');
      assert.ok(layout.map.height >= 180, `map too short at ${width}px`);
      assert.ok(layout.profile.bottom <= layout.center.bottom + 1, `profile escapes panel at ${width}px`);
      assert.ok(layout.center.y >= layout.header.bottom - 1, `header overlap at ${width}px`);
      if (width > 1120) {
        assert.ok(layout.mission.right <= layout.center.x, 'mission overlaps map');
        assert.ok(layout.center.right <= layout.telemetry.x, 'telemetry overlaps map');
      } else {
        assert.ok(layout.mission.y >= layout.center.bottom, 'map should precede mission on smaller screens');
        assert.ok(layout.telemetry.y >= layout.center.bottom, 'map should precede telemetry on smaller screens');
      }
      if (screenshots) await page.screenshot({ path: path.join(artifacts, `console-${width}-${height}.png`), fullPage: true });
      console.log(`PASS: ${width}×${height} console layout and NASA-inspired palette`);
    }

    await page.setViewportSize({ width: 1440, height: 900 });
    await page.locator('#btnNext').click();
    assert.equal(await page.locator('#stepNow').innerText(), '1');
    await page.locator('#btnStart').click();
    await page.locator('[data-mode="visor"]').click();
    assert.equal(await page.locator('#streetPrev').isDisabled(), true);
    assert.match(await page.locator('#vStationTitle').innerText(), /ROVER BASE|READY FOR EVA/);
    assert.ok(await page.locator('#visorPano').isVisible());
    await page.locator('#streetNext').click();
    await page.locator('#visorPhotoBtn').click();
    assert.ok(await page.locator('#visorPhoto').isVisible(), 'single-frame Mastcam fallback missing');
    await page.locator('[data-mode="console"]').click();
    await page.locator('.station-card').nth(4).click();
    await page.locator('[data-mode="visor"]').click();
    await page.waitForSelector('#visorMosaicWrap', { state: 'visible' });
    await page.waitForFunction(() => document.querySelector('#visorMosaicCanvas').width > 0);
    await page.waitForFunction(() => {
      const canvas = document.querySelector('#visorMosaicCanvas');
      const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let painted = 0;
      for (let i = 0; i < data.length; i += 4) if (data[i] + data[i + 1] + data[i + 2] > 60) painted++;
      return painted > 120000;
    });
    for (const width of [1440, 1024, 390]) {
      await page.setViewportSize({ width, height: 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      assert.ok(await page.locator('#visorDock').isVisible());
      if (screenshots) await page.screenshot({ path: path.join(artifacts, `visor-${width}.png`), fullPage: true });
    }
    console.log('PASS: timeline controls, visor navigation, single frame and stitched mosaic');

    // Second landmark (ALT-10) must load its own stitched mosaic.
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.locator('[data-mode="console"]').click();
    await page.locator('.station-card').nth(9).click();
    await page.locator('[data-mode="visor"]').click();
    await page.waitForSelector('#visorMosaicWrap', { state: 'visible' });
    await page.waitForFunction(() => document.querySelector('#vImageId').textContent === 'MOSAIC-ALT-10');
    await page.waitForFunction(() => {
      const canvas = document.querySelector('#visorMosaicCanvas');
      const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let painted = 0;
      for (let i = 0; i < data.length; i += 4) if (data[i] + data[i + 1] + data[i + 2] > 60) painted++;
      return painted > 120000;
    });
    console.log('PASS: both landmark stitched mosaics (ALT-05, ALT-10)');

    await page.setViewportSize({ width: 1440, height: 900 });
    await page.locator('[data-mode="console"]').click();
    await page.locator('#flareBtn').click();
    assert.ok(await page.locator('#flashBanner').isVisible());
    assert.equal(await page.locator('body').evaluate((el) => el.classList.contains('spe-active')), true);
    await page.locator('#flareBtn').click();
    assert.equal(await page.locator('#flashBanner').isVisible(), false);
    console.log('PASS: SPE activation and stand down');

    // First-person certified-route EVA walk: keyboard movement, drag-look,
    // touch pad, exit, then a full accelerated station arrival and dwell.
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.locator('[data-mode="console"]').click();
    await page.locator('#btnStart').click();
    await page.locator('[data-mode="visor"]').click();
    assert.ok(await page.locator('#walkToggle').isVisible());
    await page.locator('#walkToggle').click();
    await page.waitForSelector('body.walk-active');
    assert.ok(await page.locator('#walkHud').isVisible());
    assert.equal(await page.locator('#walkToggleLabel').innerText(), 'EXIT WALK');
    assert.match(await page.locator('#walkDistance').innerText(), /^0 \/ 647 m$/);
    assert.match(await page.locator('#walkStations').innerText(), /^0 \/ 10$/);
    assert.match(await page.locator('#walkNext').innerText(), /ALT-01/);

    const walkDistance = async () => Number((await page.locator('#walkDistance').innerText()).match(/(\d+)\s*\//)[1]);
    const beforeMove = await walkDistance();
    await page.keyboard.down('w');
    await page.waitForTimeout(1600);
    assert.ok(Number.parseFloat(await page.locator('#walkSpeed').innerText()) > 0.1, 'walk speed should be live');
    await page.waitForTimeout(1600);
    await page.keyboard.up('w');
    await page.waitForTimeout(200);
    const afterMove = await walkDistance();
    assert.ok(afterMove > beforeMove, `keyboard walk should advance route (${beforeMove} -> ${afterMove})`);

    const headingBefore = await page.locator('#pipHeading').innerText();
    await page.mouse.move(720, 430);
    await page.mouse.down();
    await page.mouse.move(880, 430, { steps: 8 });
    await page.mouse.up();
    await page.waitForTimeout(250);
    assert.notEqual(await page.locator('#pipHeading').innerText(), headingBefore, 'drag should look around');

    await page.setViewportSize({ width: 390, height: 900 });
    await page.waitForTimeout(300);
    assert.ok(await page.locator('#walkPad').isVisible(), 'touch pad should be visible on narrow screens');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    const beforeTouch = await walkDistance();
    await page.locator('#walkPad .up').dispatchEvent('pointerdown', { pointerId: 7, pointerType: 'touch', bubbles: true });
    await page.waitForTimeout(1800);
    await page.locator('#walkPad .up').dispatchEvent('pointerup', { pointerId: 7, pointerType: 'touch', bubbles: true });
    await page.waitForTimeout(200);
    assert.ok(await walkDistance() > beforeTouch, 'touch pad should advance the walk');

    await page.setViewportSize({ width: 1440, height: 900 });
    await page.waitForTimeout(250);
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);
    assert.equal(await page.locator('body').evaluate((el) => el.classList.contains('walk-active')), false);
    assert.equal(await page.locator('#walkHud').isVisible(), false);
    assert.equal(await page.locator('#walkToggleLabel').innerText(), 'WALK ROUTE');

    await page.locator('[data-mode="console"]').click();
    await page.locator('#btnStart').click();
    await page.locator('[data-mode="visor"]').click();
    await page.locator('#visorStepRange').evaluate((el) => {
      el.value = '1';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.locator('#walkToggle').click();
    await page.waitForSelector('#walkArrival.show');
    assert.match(await page.locator('#walkArrivalTitle').innerText(), /ALT-01/);
    assert.match(await page.locator('#walkArrivalEyebrow').innerText(), /STATION ARRIVAL/);
    assert.match(await page.locator('#walkStations').innerText(), /^0 \/ 10$/);
    await page.waitForFunction(
      () => /^1 \/ 10$/.test(document.querySelector('#walkStations').textContent.trim()),
      null,
      { timeout: 20000 }
    );
    assert.match(await page.locator('#walkArrivalEyebrow').innerText(), /DWELL COMPLETE/);
    assert.equal(await page.locator('#vStepNow').innerText(), '2');
    assert.match(await page.locator('#walkMargin').innerText(), /^\+\d/);
    await page.keyboard.press('Escape');
    console.log('PASS: EVA walk movement, drag-look, touch pad, station arrival and dwell');

    await page.goto(pathToFileURL(path.join(root, 'index.html')).href);
    await page.waitForSelector('#boot.done', { state: 'attached' });
    assert.equal(await page.locator('.station-card').count(), 10);
    console.log('PASS: offline file:// embedded bundle');
    assert.deepEqual(errors, [], 'browser runtime errors');
    console.log('PASS: no browser runtime errors');
  } finally {
    if (browser) await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
