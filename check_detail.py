import json, urllib.request

# Find the latest run
url = "https://api.github.com/repos/derekgallardo01/sonave/actions/runs?per_page=1"
req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
data = json.loads(urllib.request.urlopen(req, timeout=15).read())
run = data["workflow_runs"][0]
run_id = run["id"]

print(f"Run {run['head_sha'][:7]} - conclusion: {run.get('conclusion', '?')}")

jobs_url = f"https://api.github.com/repos/derekgallardo01/sonave/actions/runs/{run_id}/jobs"
req2 = urllib.request.Request(jobs_url, headers={"Accept": "application/vnd.github+json"})
jdata = json.loads(urllib.request.urlopen(req2, timeout=15).read())
for j in jdata.get("jobs", []):
    print(f"  Job: {j['name']} => {j.get('conclusion', 'running')}")
    for step in j.get("steps", []):
        if step.get("conclusion") in ("failure", "cancelled"):
            print(f"    FAILED: {step['name']}")
