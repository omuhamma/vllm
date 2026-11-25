"""Snapshot / diff the vLLM spec-decode counters from /metrics.
usage: python3 tools/k31_spec_metrics.py snap <out.json> | diff <a.json> <b.json>"""
import json, re, sys, urllib.request
def snap(port=8131):
    t = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=30).read().decode()
    d = {}
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode") and not l.startswith("#"):
            k, v = l.rsplit(" ", 1)
            k = re.sub(r'(engine|model_name)="[^"]*",?', "", k).replace("{}", "")
            d[k] = d.get(k, 0) + float(v)
    return d
if sys.argv[1] == "snap":
    json.dump(snap(), open(sys.argv[2], "w"), indent=1)
else:
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    D = {k: b[k] - a.get(k, 0) for k in b}
    for k in sorted(D): print(k, D[k])
    nd = next((v for k, v in D.items() if k.startswith("vllm:spec_decode_num_drafts")), 0)
    nt = next((v for k, v in D.items() if k.startswith("vllm:spec_decode_num_draft_tokens")), 0)
    na = next((v for k, v in D.items() if k.startswith("vllm:spec_decode_num_accepted_tokens_total")), 0)
    pos = sorted(((int(re.search(r'position="(\d+)"', k).group(1)), v) for k, v in D.items() if "per_pos" in k and "position=" in k))
    if nd:
        print(f"drafts {nd:.0f} draft_tokens {nt:.0f} accepted {na:.0f}")
        print(f"mean acceptance length {1 + na / nd:.4f}   acceptance rate {na / nt:.4f}")
        print("per-position acceptance:", ", ".join(f"{p}: {v / nd:.4f}" for p, v in pos))
