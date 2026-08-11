import json, urllib.request

# Check the latest run (331309b)
url = "https://api.github.com/repos/derekgallardo01/sonave/actions/runs?per_page=1"
req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
data = json.loads(urllib.request.urlopen(req, timeout=15).read())
run = data["workflow_runs"][0]
run_id = run["id"]

jobs_url = f"https://api.github.com/repos/derekgallardo01/sonave/actions/runs/{run_id}/jobs"
req2 = urllib.request.Request(jobs_url, headers={"Accept": "application/vnd.github+json"})
jdata = json.loads(urllib.request.urlopen(req2, timeout=15).read())
for j in jdata.get("jobs", []):
    print(f"Job: {j['name']} | Conclusion: {j.get('conclusion', '?')}")
    for step in j.get("steps", []):
        print(f"  Step {step['number']}: {step['name']} => {step.get('conclusion', 'running')}")
