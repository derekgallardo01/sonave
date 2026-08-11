import json, urllib.request
url = "https://api.github.com/repos/derekgallardo01/sonave/actions/runs?per_page=3"
req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
data = json.loads(urllib.request.urlopen(req, timeout=15).read())
for r in data.get("workflow_runs", []):
    print(f"{r['status']:12} | {str(r.get('conclusion','?')):8} | {r['head_sha'][:7]} | {r['name']}")
