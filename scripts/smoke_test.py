"""
Smoke test a deployed API: wait until it runs the expected commit, then check
that predictions and input validation behave as they do in the test suite.

Usage:
    python scripts/smoke_test.py https://telco-fastapi.onrender.com [expected_git_sha]

Standard library only, so CI can run it without installing anything.
Exits non-zero (failing the CI job) if any check fails.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1].rstrip("/")
EXPECTED_SHA = sys.argv[2] if len(sys.argv) > 2 else None
WAIT_SECONDS = 15 * 60  # Render free tier: deploy plus cold start can take several minutes

CUSTOMER = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 350.75,
}


def request(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def wait_for_deploy():
    deadline = time.time() + WAIT_SECONDS
    while time.time() < deadline:
        try:
            status, body = request("/version")
            live = body.get("git_sha") if status == 200 else None
            if EXPECTED_SHA is None or live == EXPECTED_SHA:
                print(f"Live version: {live}")
                return
            print(f"Live version {live}, waiting for {EXPECTED_SHA}...")
        except Exception as e:  # service restarting or waking up
            print(f"Not reachable yet ({e.__class__.__name__}), retrying...")
        time.sleep(15)
    sys.exit(f"FAIL: {EXPECTED_SHA} was not live after {WAIT_SECONDS // 60} minutes")


def main():
    wait_for_deploy()
    failures = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")
        if not ok:
            failures.append(name)

    status, body = request("/")
    check("health check", status == 200 and body == {"status": "ok"})

    probs = {}
    for contract in ["Month-to-month", "One year", "Two year"]:
        status, body = request("/predict", dict(CUSTOMER, Contract=contract))
        probs[contract] = body.get("probability") if status == 200 else None
    check(
        "longer contracts lower churn risk",
        None not in probs.values() and probs["Month-to-month"] > probs["One year"] > probs["Two year"],
        str(probs),
    )

    status, body = request("/predict", dict(CUSTOMER, Contract="Two Yr"))
    check("invalid value rejected with 422", status == 422, str(body))

    status, body = request("/model-stats")
    check("model stats available", status == 200 and body.get("threshold") == 0.35)

    status, body = request("/feature-importance")
    check("feature importance available", status == 200 and len(body.get("feature_importance", [])) == 30)

    if failures:
        sys.exit(f"{len(failures)} smoke test(s) failed: {failures}")
    print("All smoke tests passed")


if __name__ == "__main__":
    main()
