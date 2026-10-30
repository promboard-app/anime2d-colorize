// 앱 화면에서 예제 워크플로우를 실제로 돌리고 화면을 찍는다(README 캡처 · 워크플로우 썸네일 재료).
// 격리 PromBoard 서버(예: 127.0.0.1:27196)에 이 플러그인과 팩 플러그인이 켜지고 실행 준비가 끝나 있어야 한다.
//   node docs/scripts/app_e2e.cjs <서버 주소> <PromBoard 저장소 node_modules/playwright> <브라우저 프로필 폴더> <출력 폴더> [ko|en]
// 새 프로젝트 → 빈 프로젝트 → 워크플로우 불러오기 → 에셋 › 플러그인 카드 3장을 소스 노드에 끌어 놓기 → 팩 노드 ▷ → 채색 노드 ▷.
const [base, playwrightPath, profile, outDir, lang = 'ko'] = process.argv.slice(2);
const { chromium } = require(playwrightPath);
const path = require('path');
const T = lang === 'en'
  ? { later: /Maybe later|Not now|Later/, projects: 'My Projects', newProject: 'New project', blank: 'Blank project', load: 'Load workflow', assets: 'Assets', plugins: 'Plugins', run: 'Run' }
  : { later: /다음에 할게요/, projects: '내 프로젝트', newProject: '새 프로젝트', blank: '빈 프로젝트', load: '워크플로우 불러오기', assets: '에셋', plugins: '플러그인', run: '실행' };
const WORKFLOW = '이어진 선화 두 장 → 중간 프레임 → 선화 채색';
const CARDS = ['예제 — 이어진 선화 시작 프레임', '예제 — 이어진 선화 끝 프레임', '예제 — 캐릭터 설정화'];

async function nodesOnCanvas(page) {
  return page.locator('[data-node-id]').evaluateAll((els) => els.map((e) => {
    const r = e.getBoundingClientRect();
    return { id: e.getAttribute('data-node-id'), x: r.x, y: r.y, text: e.innerText };
  }));
}

async function runAndWait(page, id, label) {
  const node = page.locator(`[data-node-id="${id}"]`);
  await node.hover();
  await page.waitForTimeout(800);
  const t0 = Date.now();
  await node.getByRole('button', { name: T.run }).first().click({ force: true });
  let seenRunning = false;
  for (let i = 0; i < 400; i++) {
    await page.waitForTimeout(3000);
    const txt = await node.innerText();
    const running = /\d\/3 ·/.test(txt);
    if (running) seenRunning = true;
    if (/실패|failed/i.test(txt) && !running) throw new Error(`${label} failed: ${txt.slice(0, 300)}`);
    if (seenRunning && !running) { console.log(label, 'done', Math.round((Date.now() - t0) / 1000), 's'); return; }
  }
  throw new Error(`${label} timeout`);
}

(async () => {
  const ctx = await chromium.launchPersistentContext(profile, { headless: true, viewport: { width: 1600, height: 1000 }, args: ['--disable-gpu'] });
  const page = ctx.pages()[0] || (await ctx.newPage());
  await page.goto(`${base}/home`, { waitUntil: 'load' });
  await page.evaluate((l) => {
    try {
      const raw = localStorage.getItem('ui-storage');
      const s = raw ? JSON.parse(raw) : { state: {}, version: 0 };
      s.state = { ...(s.state || {}), language: l };
      localStorage.setItem('ui-storage', JSON.stringify(s));
    } catch (e) { /* 언어는 처음 값으로 둔다 */ }
  }, lang);
  await page.reload({ waitUntil: 'load' });
  await page.waitForTimeout(5000);
  await page.getByText(T.projects).first().click();
  await page.waitForTimeout(1500);
  await page.getByText(T.newProject, { exact: true }).first().click();
  await page.waitForTimeout(1500);
  await page.getByText(T.blank).first().click();
  await page.waitForTimeout(5000);
  const later = page.getByText(T.later);
  if (await later.count()) await later.first().click();
  await page.waitForTimeout(800);
  await page.getByText(T.load).first().click();
  await page.waitForTimeout(2000);
  await page.getByText(WORKFLOW).first().click();
  await page.waitForTimeout(4000);
  const nodes = await nodesOnCanvas(page);
  const sources = nodes.filter((n) => !/애니메이션/.test(n.text)).sort((a, b) => a.x - b.x || a.y - b.y);
  const fill = nodes.find((n) => /프레임 사이 채우기/.test(n.text));
  const colorize = nodes.find((n) => /선화 채색/.test(n.text));
  // 소스 셋: 왼쪽 위 = 시작, 왼쪽 아래 = 끝, 가운데 아래 = 설정화(워크플로우 배치 그대로)
  const left = sources.filter((n) => n.x === sources[0].x).sort((a, b) => a.y - b.y);
  const sheet = sources.find((n) => n.x !== sources[0].x);
  const targets = [left[0].id, left[1].id, sheet.id];
  await page.getByText(T.assets, { exact: true }).first().click();
  await page.waitForTimeout(1200);
  await page.getByText(T.plugins, { exact: true }).last().click();
  await page.waitForTimeout(2500);
  for (let i = 0; i < 3; i++) {
    const card = page.locator(`[draggable="true"]:has(img[alt="${CARDS[i]}"])`).first();
    const target = page.locator(`[data-node-id="${targets[i]}"]`);
    const box = await target.boundingBox();
    await card.dragTo(target, { targetPosition: { x: Math.round(box.width) - 12, y: Math.round(box.height / 2) } });
    await page.waitForTimeout(2500);
  }
  await page.getByText(T.assets, { exact: true }).first().click(); // 패널을 노드 목록 쪽으로 돌려 캔버스를 덜 가린다
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(outDir, `app-${lang}-before.png`) });
  await runAndWait(page, fill.id, 'fill');
  await runAndWait(page, colorize.id, 'colorize');
  await page.waitForTimeout(3000);
  await page.mouse.move(5, 995);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(outDir, `app-${lang}.png`) });
  console.log('url', page.url());
  await ctx.close();
})().catch((e) => { console.error(e); process.exit(1); });
