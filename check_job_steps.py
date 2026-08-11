import json, urllib.request, zipfile, io

# GitHub API needs no auth for public repos, but logs might redirect to an authenticated URL
run_id = 31540790935
url = f"https://api.github.com/repos/derekgallardo01/sonave/actions/runs/{run_id}/jobs"
req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
data = json.loads(urllib.request.urlopen(req, timeout=15).read())

for j in data.get("jobs", []):
    if j.get("conclusion") == "failure":
        print(f"Failed job: {j['name']}")
        # Get steps
        for step in j.get("steps", []):
            if step.get("conclusion") == "failure":
                print(f"  Failed step: {step['name']} (number {step['number']})")
                print(f"  Status: {step['status']}")
