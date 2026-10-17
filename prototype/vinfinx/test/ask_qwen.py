import json, urllib.request
URL = "http://127.0.0.1:8061/vinfinx-api/v1/chat/completions"
history = []
while True:
    q = input("\nyou> ").strip()
    if q in {"/quit", "/q"}: break
    if q == "/reset": history = []; print("(history cleared)"); continue
    history.append({"role": "user", "content": q})
    body = json.dumps({"messages": history[-12:], "max_tokens": 400}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json",
                                             "Origin": "http://127.0.0.1:4173"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    a = d["choices"][0]["message"]["content"]
    history.append({"role": "assistant", "content": a})
    print(f"\nqwen> {a}\n[{d['usage']}]")
