"""COS60011 prototype performance tests.

Place this file alongside app.py and model_loader.py in the app/ folder.
Run: python app/prototype_performance_test.py

Inference measurements are LOCAL (direct model_loader calls), not Streamlit Cloud
end-to-end inference measurements. Homepage measurements use a real Playwright Chromium browser against the
deployed website, not model predictions. HTTP probes remain diagnostic only.
"""
import csv
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Barrier
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen, build_opener, HTTPRedirectHandler
from urllib.parse import urljoin, urlparse

from model_loader import load_model, predict_emotion

WEBSITE = "https://cos6011-technology-design-project-mg3-emotion-classifier.streamlit.app/"
TEXT = "The company reported strong earnings and its share price increased significantly."
ROUNDS = 3
HOMEPAGE_REQUESTS = 20
BROWSER_ROUNDS = 3
BROWSER_TIMEOUT_MS = 60000
HEALTH_URL = WEBSITE.rstrip("/") + "/_stcore/health"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "prototype_testing"
LABELS = {"anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"}


def timed_prediction(barrier=None):
    if barrier is not None:
        barrier.wait(timeout=30)
    start = time.perf_counter()
    try:
        result = predict_emotion(TEXT)
        valid = (result["emotion"] in LABELS and
                 set(result["probabilities"]) == LABELS and
                 abs(sum(result["probabilities"].values()) - 1.0) < 0.01)
        error = "" if valid else "Unexpected prediction output"
        return {"seconds": time.perf_counter() - start,
                "success": valid, "error": error, "http_status": "", "redirect_location": ""}
    except Exception as exc:
        return {"seconds": time.perf_counter() - start,
                "success": False, "error": str(exc), "http_status": "", "redirect_location": ""}


def run_inference_case(users, rounds):
    samples = []
    start = time.perf_counter()
    for _ in range(rounds):
        barrier = Barrier(users) if users > 1 else None
        with ThreadPoolExecutor(max_workers=users) as pool:
            futures = [pool.submit(timed_prediction, barrier) for _ in range(users)]
            samples.extend(f.result() for f in as_completed(futures))
    wall = time.perf_counter() - start
    return samples, wall


class NoRedirect(HTTPRedirectHandler):
    """Inspect redirect responses without following auth/session loops."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


NO_REDIRECT_OPENER = build_opener(NoRedirect)


def http_probe(url, expect_health=False):
    """Measure one HTTP request; distinguish redirects from genuine failures."""
    start = time.perf_counter()
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (COS60011 academic test)"})
    try:
        with NO_REDIRECT_OPENER.open(request, timeout=15) as response:
            status = response.status
            body = response.read(1024)
            if expect_health:
                valid = status == 200 and body.strip().lower() == b"ok"
            else:
                valid = status == 200
            return {"seconds": time.perf_counter() - start,
                    "success": valid, "error": "" if valid else f"Unexpected HTTP {status} / response",
                    "http_status": status, "redirect_location": ""}
    except HTTPError as exc:
        location = exc.headers.get("Location", "")
        # A redirect is an inconclusive accessibility check, not a failed inference.
        return {"seconds": time.perf_counter() - start, "success": False,
                "error": f"HTTP {exc.code}" + (" (redirect; possible auth/session)" if 300 <= exc.code < 400 else ""),
                "http_status": exc.code, "redirect_location": urljoin(url, location) if location else ""}
    except (URLError, TimeoutError, OSError) as exc:
        return {"seconds": time.perf_counter() - start, "success": False,
                "error": str(exc), "http_status": "", "redirect_location": ""}




def browser_homepage_probe(attempt=1):
    """Automated readiness: inspect DOM text across every tab and frame, not brittle CSS selectors."""
    start = time.perf_counter()
    diagnostic_dir = OUTPUT_DIR / "browser_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"seconds": 0.0, "success": False, "error": "Install Playwright and Chromium",
                "http_status": "", "redirect_location": ""}

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False)
            try:
                context = browser.new_context(viewport={"width": 1440, "height": 900})
                page = context.new_page()
                response = page.goto(WEBSITE, wait_until="domcontentloaded", timeout=BROWSER_TIMEOUT_MS)
                deadline = time.perf_counter() + BROWSER_TIMEOUT_MS / 1000
                last_observation = ""
                while time.perf_counter() < deadline:
                    # Inspect all open tabs and frames: Streamlit can embed its app
                    # or navigate while establishing its session.
                    for tab in context.pages:
                        for frame in tab.frames:
                            try:
                                state = frame.evaluate("""() => {
                                    const body = document.body;
                                    if (!body) return {text: '', buttons: 0, inputs: 0};
                                    const text = body.innerText || '';
                                    const buttons = [...document.querySelectorAll('button')];
                                    const inputs = [...document.querySelectorAll('textarea,input')];
                                    return {
                                        text: text.slice(0, 3000),
                                        buttons: buttons.filter(b => /Analyse Emotion/i.test(b.innerText || b.getAttribute('aria-label') || '')).length,
                                        inputs: inputs.length
                                    };
                                }""")
                                # Require both the actual classifier action and an input.
                                if state["buttons"] > 0 and state["inputs"] > 0:
                                    return {"seconds": time.perf_counter() - start,
                                            "success": True, "error": "",
                                            "http_status": response.status if response else "",
                                            "redirect_location": tab.url}
                                last_observation = (
                                    f"tab={tab.url}; frame={frame.url}; "
                                    f"buttons={state['buttons']}; inputs={state['inputs']}; "
                                    f"text={state['text'][:140]!r}"
                                )
                            except Exception as exc:
                                last_observation = f"Frame inspection: {type(exc).__name__}: {str(exc)[:120]}"
                    page.wait_for_timeout(500)

                screenshot = diagnostic_dir / f"homepage_failure_{attempt}.png"
                page.screenshot(path=str(screenshot), full_page=True, timeout=10000)
                return {"seconds": time.perf_counter() - start, "success": False,
                        "error": f"DOM readiness not detected; {last_observation}; screenshot={screenshot}",
                        "http_status": response.status if response else "",
                        "redirect_location": page.url}
            finally:
                browser.close()
    except Exception as exc:
        return {"seconds": time.perf_counter() - start, "success": False,
                "error": f"{type(exc).__name__}: {str(exc)[:350]}",
                "http_status": "", "redirect_location": ""}

def summarize(name, mode, users, samples, wall):
    latencies = [s["seconds"] for s in samples]
    successes = sum(s["success"] for s in samples)
    return {"test_case": name, "measurement_location": mode, "concurrent_users": users,
            "requests": len(samples), "successes": successes,
            "failures": len(samples) - successes,
            "error_rate_percent": round(100 * (len(samples) - successes) / len(samples), 2),
            "average_seconds": round(statistics.mean(latencies), 4),
            "max_seconds": round(max(latencies), 4),
            "throughput_successes_per_second": round(successes / wall, 4) if wall else 0,
            "elapsed_seconds": round(wall, 4)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading BERT model locally (warm-up excluded from timed results)...")
    load_model()
    warmup = timed_prediction()
    if not warmup["success"]:
        raise RuntimeError(f"Warm-up failed: {warmup['error']}")
    results = []
    details = []
    for users in (1, 5, 10):
        name = f"Local inference - {users} user(s)"
        print(f"Testing {name}, {ROUNDS} round(s)...")
        samples, wall = run_inference_case(users, ROUNDS)
        results.append(summarize(name, "Local Python model inference", users, samples, wall))
        details.extend({"test_case": name, "request_number": i, **s} for i, s in enumerate(samples, 1))
    print("Checking deployed homepage redirect status (one diagnostic request)...")
    homepage_sample = http_probe(WEBSITE)
    results.append(summarize("Homepage HTTP GET diagnostic", "Streamlit Cloud HTTP", 1,
                             [homepage_sample], homepage_sample["seconds"]))
    details.append({"test_case": "Homepage HTTP GET diagnostic", "request_number": 1,
                    **homepage_sample})
    if homepage_sample["redirect_location"]:
        print("Homepage redirected to:", homepage_sample["redirect_location"])
    print(f"Testing deployed Streamlit health endpoint: {HOMEPAGE_REQUESTS} sequential requests...")
    start = time.perf_counter()
    health_samples = [http_probe(HEALTH_URL, expect_health=True) for _ in range(HOMEPAGE_REQUESTS)]
    wall = time.perf_counter() - start
    results.append(summarize("Deployed health endpoint HTTP GET", "Streamlit Cloud HTTP", 1,
                             health_samples, wall))
    details.extend({"test_case": "Deployed health endpoint HTTP GET", "request_number": i, **sample}
                   for i, sample in enumerate(health_samples, 1))
    print(f"Testing deployed homepage in Chromium: {BROWSER_ROUNDS} browser launches...")
    browser_start = time.perf_counter()
    browser_samples = [browser_homepage_probe(i) for i in range(1, BROWSER_ROUNDS + 1)]
    browser_wall = time.perf_counter() - browser_start
    results.append(summarize("Deployed homepage browser readiness",
                             "Streamlit Cloud Chromium browser", 1, browser_samples, browser_wall))
    details.extend({"test_case": "Deployed homepage browser readiness",
                    "request_number": i, **sample}
                   for i, sample in enumerate(browser_samples, 1))
    for i, sample in enumerate(browser_samples, 1):
        if not sample["success"]:
            print(f"Browser test {i} issue: {sample['error']}")
    for filename, rows in (("performance_summary.csv", results), ("performance_requests.csv", details)):
        path = OUTPUT_DIR / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {path}")
    for result in results:
        print(f"{result['test_case']}: average={result['average_seconds']}s, "
              f"max={result['max_seconds']}s, failures={result['failures']}, "
              f"throughput={result['throughput_successes_per_second']}/s")
    print("NOTE: Inference is measured LOCALLY, not via the deployed Streamlit UI.")
    print("NOTE: Health endpoint checks HTTP availability only; it does NOT run model inference.")
    print("NOTE: Homepage redirects are diagnostic, not evidence of failed predictions.")
    print("NOTE: Browser readiness measures time to visible UI, not BERT inference or load throughput.")


if __name__ == "__main__":
    main()
