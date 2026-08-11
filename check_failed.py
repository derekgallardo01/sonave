import json, urllib.request
run_id = None
# Find the failed run for a21e215
url = "https://api.github.com/repos/derekgallardo01/sonave/actions/runs?per_page=3"
req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
data = json.loads(urllib.request.urlopen(req, timeout=15).read())
for r in data.get("workflow_runs", []):
    if r["head_sha"].startswith("a21e215"):
        run_id = r["id"]
        break

if run_id:
    jobs_url = f"https://api.github.com/repos/derekgallardo01/sonave/actions/runs/{run_id}/jobs"
    req2 = urllib.request.Request(jobs_url, headers={"Accept": "application/vnd.github+json"})
    jdata = json.loads(urllib.request.urlopen(req2, timeout=15).read())
    for j in jdata.get("jobs", []):
        print(f"Job: {j['name']} | Status: {j['status']} | Conclusion: {j.get('conclusion', '?')}")
        for step in j.get("steps", []):
            if step.get("conclusion") == "failure":
                print(f"  FAILED STEP: {step['name']} (number {step['number']})")
else:
    print("Run not found")
