import json, urllib.request
url = "https://api.github.com/repos/derekgallardo01/sonave/actions/runs?per_page=1&status=completed"
req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
data = json.loads(urllib.request.urlopen(req, timeout=15).read())
run = data["workflow_runs"][0]
print(f"Run ID: {run['id']}")
print(f"Conclusion: {run['conclusion']}")
print(f"URL: {run['html_url']}")

# Fetch jobs
jobs_url = run["jobs_url"]
req2 = urllib.request.Request(jobs_url, headers={"Accept": "application/vnd.github+json"})
jdata = json.loads(urllib.request.urlopen(req2, timeout=15).read())
for j in jdata.get("jobs", []):
    print(f"  Job: {j['name']} | Status: {j['status']} | Conclusion: {j.get('conclusion', '?')}")
    if j.get("conclusion") == "failure":
        # Get logs URL
        print(f"  Logs: {j['html_url']}")
