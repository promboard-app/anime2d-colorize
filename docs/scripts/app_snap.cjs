// 이미 돌린 보드를 화면 언어별로 다시 찍는다(선택 해제 · 에셋 › 플러그인 카드가 보이게).
//   node docs/scripts/app_snap.cjs <보드 주소> <node_modules/playwright> <프로필> <출력 png> [ko|en]
const [url, playwrightPath, profile, out, lang = 'ko'] = process.argv.slice(2);
const { chromium } = require(playwrightPath);
(async () => {
  const ctx = await chromium.launchPersistentContext(profile, { headless: true, viewport: { width: 1600, height: 1000 }, args: ['--disable-gpu'] });
  const page = ctx.pages()[0] || (await ctx.newPage());
  await page.goto(url, { waitUntil: 'load' });
  await page.evaluate((l) => {
    try {
      const raw = localStorage.getItem('ui-storage');
      const s = raw ? JSON.parse(raw) : { state: {}, version: 0 };
      s.state = { ...(s.state || {}), language: l };
      localStorage.setItem('ui-storage', JSON.stringify(s));
    } catch (e) { /* 처음 값 */ }
  }, lang);
  await page.reload({ waitUntil: 'load' });
  await page.waitForTimeout(6000);
  const later = page.getByText(/다음에 할게요|Maybe later|Not now|Later/);
  if (await later.count()) await later.first().click();
  await page.waitForTimeout(800);
  await page.getByText(lang === 'en' ? 'Assets' : '에셋', { exact: true }).first().click();
  await page.waitForTimeout(1000);
  await page.getByText(lang === 'en' ? 'Plugins' : '플러그인', { exact: true }).last().click();
  await page.waitForTimeout(2000);
  await page.mouse.click(1400, 930); // 빈 캔버스 — 선택 해제
  await page.waitForTimeout(800);
  // 캔버스를 오른쪽으로 끌어 왼쪽 패널에 가린 소스 노드가 보이게(손 도구 = 스페이스 누른 채 끌기)
  await page.keyboard.down('Space');
  await page.mouse.move(1100, 880);
  await page.mouse.down();
  await page.mouse.move(1260, 900, { steps: 12 });
  await page.mouse.up();
  await page.keyboard.up('Space');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: out });
  await ctx.close();
})().catch((e) => { console.error(e); process.exit(1); });
