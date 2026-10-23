"""COS60011: deployed Streamlit end-to-end browser load test.

Run from project root: python app/deployed_streamlit_load_test.py
Requirements: pip install playwright ; python -m playwright install chromium
Measures click-to-visible-result latency, NOT local BERT inference.
Three rounds each for 1, 5, and 10 independent browser contexts.
"""
import asyncio
import csv
import statistics
import time
from pathlib import Path

from playwright.async_api import async_playwright

WEBSITE = "https://cos6011-technology-design-project-mg3-emotion-classifier.streamlit.app/"
TEXT = "The company reported strong earnings and its share price increased significantly."
CONCURRENCY = (1, 5, 10)
ROUNDS = 3
READY_TIMEOUT_MS = 90000
PREDICTION_TIMEOUT_MS = 90000
HEADLESS = False  # Visible browser worked reliably with this deployment.
OUTPUT = Path(__file__).resolve().parent.parent / "results" / "prototype_testing"

# Find button in the rendered DOM rather than relying on accessibility-name matching.
BUTTON_EXISTS = """() => [...document.querySelectorAll('button')].some(
  b => /Analyse Emotion/i.test(b.innerText || b.getAttribute('aria-label') || '')
)"""
BUTTON_CLICK = """() => {
  const b = [...document.querySelectorAll('button')].find(
    b => /Analyse Emotion/i.test(b.innerText || b.getAttribute('aria-label') || '')
  );
  if (!b || b.disabled) return false;
  b.click(); return true;
}"""
RESULT_EXISTS = """() => {
  const text = document.body?.innerText || '';
  return text.includes('Analysis completed!') && text.includes('Predicted Emotion');
}"""

async def find_classifier_frame(page):
    deadline = time.perf_counter() + READY_TIMEOUT_MS / 1000
    while time.perf_counter() < deadline:
        for frame in page.frames:
            try:
                has_input = await frame.locator('textarea').count() > 0
                has_button = await frame.evaluate(BUTTON_EXISTS)
                if has_input and has_button:
                    return frame
            except Exception:
                pass
        await asyncio.sleep(0.4)
    raise TimeoutError('Classifier textarea/button not found within readiness timeout')

async def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    details = []
    summaries = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS)
        try:
            for users in CONCURRENCY:
                all_records = []
                total_wall = 0.0
                for round_number in range(1, ROUNDS + 1):
                    print(f'Preparing {users} browser session(s), round {round_number}/{ROUNDS}...')
                    start_event = asyncio.Event()
                    prepared = [asyncio.Event() for _ in range(users)]
                    async def prepared_user(i):
                        # Use an explicit per-user readiness event so requests are synchronized.
                        return await run_prepared_user(browser, users, round_number, i + 1,
                                                       start_event, prepared[i])
                    tasks = [asyncio.create_task(prepared_user(i)) for i in range(users)]
                    try:
                        await asyncio.wait_for(asyncio.gather(*(p.wait() for p in prepared)),
                                               timeout=(READY_TIMEOUT_MS / 1000) + 20)
                    except asyncio.TimeoutError:
                        print('Some sessions did not prepare before timeout; releasing others.')
                    wall_start = time.perf_counter()
                    start_event.set()
                    records = await asyncio.gather(*tasks)
                    wall = time.perf_counter() - wall_start
                    total_wall += wall
                    all_records.extend(records)
                    details.extend(records)
                    passed = sum(r['success'] for r in records)
                    print(f'  Completed: {passed}/{users}; round wall time {wall:.2f}s')
                latencies = [float(r['prediction_seconds']) for r in all_records if r['success']]
                successes = len(latencies)
                total = len(all_records)
                summaries.append({
                    'test_case': f'Deployed prediction - {users} concurrent user(s)',
                    'concurrent_users': users, 'requests': total,
                    'successes': successes, 'failures': total - successes,
                    'error_rate_percent': round(100 * (total - successes) / total, 2),
                    'average_prediction_seconds': round(statistics.mean(latencies), 4) if latencies else '',
                    'maximum_prediction_seconds': round(max(latencies), 4) if latencies else '',
                    'throughput_predictions_per_second': round(successes / total_wall, 4) if total_wall else 0,
                    'total_submission_wall_seconds': round(total_wall, 4),
                })
        finally:
            await browser.close()
    for name, rows in [('deployed_prediction_requests.csv', details),
                       ('deployed_prediction_summary.csv', summaries)]:
        dest = OUTPUT / name
        with dest.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('Saved:', dest)
    print('All predictions: ', sum(s['successes'] for s in summaries), '/',
          sum(s['requests'] for s in summaries))
    print('NOTE: Browser-context concurrency, not guaranteed simultaneous server execution.')
    print('NOTE: Click-to-result latency excludes homepage readiness; throughput includes wait to finish each round.')

async def run_prepared_user(browser, users, round_number, user_number, start_event, prepared_event):
    context = None
    page = None
    record = {'scenario_users': users, 'round': round_number, 'user': user_number,
              'ready_seconds': '', 'prediction_seconds': '', 'success': False,
              'error': '', 'final_url': ''}
    try:
        context = await browser.new_context(viewport={'width': 1440, 'height': 900})
        page = await context.new_page()
        start = time.perf_counter()
        await page.goto(WEBSITE, wait_until='domcontentloaded', timeout=READY_TIMEOUT_MS)
        frame = await find_classifier_frame(page)
        record['ready_seconds'] = round(time.perf_counter() - start, 4)
        # Streamlit text_area commits its widget value on blur (or Ctrl+Enter).
        # Filling the DOM alone may leave the Python session with an empty string.
        textarea = frame.locator('textarea').first
        await textarea.fill(TEXT)
        await textarea.press('Tab')  # Blur triggers Streamlit's widget update/rerun.
        # Allow Streamlit's WebSocket round-trip to finish before submission.
        await frame.wait_for_function(
            '''expected => {
                const ta = document.querySelector('textarea');
                return ta && ta.value === expected;
            }''', arg=TEXT, timeout=READY_TIMEOUT_MS
        )
        await asyncio.sleep(1.5)
        # Reacquire after Streamlit's rerender, and verify text remains populated.
        frame = await find_classifier_frame(page)
        if await frame.locator('textarea').first.input_value() != TEXT:
            raise RuntimeError('Financial text did not persist after Streamlit update')
        prepared_event.set()
        await start_event.wait()
        start = time.perf_counter()
        if not await frame.evaluate(BUTTON_CLICK):
            raise RuntimeError('Analyse Emotion button missing or disabled at submission')
        await frame.wait_for_function(RESULT_EXISTS, timeout=PREDICTION_TIMEOUT_MS, polling=200)
        record['prediction_seconds'] = round(time.perf_counter() - start, 4)
        record['success'] = True
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {str(exc)[:350]}'
        prepared_event.set()  # Ensure failed setup cannot block the whole test.
        if page:
            try:
                path = OUTPUT / 'browser_diagnostics' / f'prediction_{users}users_round{round_number}_user{user_number}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(path), timeout=5000)
            except Exception:
                pass
    finally:
        if page:
            record['final_url'] = page.url
        if context:
            await context.close()
    return record

if __name__ == '__main__':
    asyncio.run(main())
