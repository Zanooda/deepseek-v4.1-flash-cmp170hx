#!/usr/bin/env python3
"""Correctness checks for a fresh DeepSeek-V4.1 sm_80 engine: coherence + needles.

1. Chat coherence: a handful of short chat completions (factual, reasoning,
   code, tool call) printed for eyeballing plus cheap automatic sanity checks
   (non-empty, no token salad: repetition ratio, printable ratio, expected
   keyword present).
2. Needle-in-haystack at several context sizes and depths (10 %, 50 %, 90 %):
   a passphrase is buried in random-word filler; the reply must contain it.
   Distinct passphrase per request so cross-request bleed is detectable, and
   the concurrent variant fires N needles at once.

Usage: bench_v41_check.py [--url http://127.0.0.1:8099] [--model dsv41]
                          [--ctx 4000,32000,128000,512000,1000000] [--conc 4]
"""
import argparse
import json
import random
import re
import sys
import threading
import time
import urllib.request

WORDS = (
    "system kernel memory buffer thread process socket packet register cache "
    "pointer allocate schedule interrupt virtual physical address translate "
    "compile execute branch predict pipeline vector matrix tensor gradient "
    "cluster network storage device driver module segment offset boundary "
    "harbor lantern meadow copper violin orchard glacier saffron thimble walnut"
).split()


def post(url, payload, timeout=7200):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return time.perf_counter() - t0, resp


def salad_score(text: str) -> dict:
    toks = re.findall(r"\w+", text.lower())
    uniq = len(set(toks)) / max(1, len(toks))
    printable = sum(c.isprintable() or c.isspace() for c in text) / max(1, len(text))
    return {"unique_ratio": round(uniq, 2), "printable": round(printable, 3), "len": len(text)}


def chat(base, model, messages, max_tokens=300, **kw):
    dt, r = post(f"{base}/v1/chat/completions",
                 {"model": model, "messages": messages, "max_tokens": max_tokens,
                  "temperature": 0, **kw})
    msg = r["choices"][0]["message"]
    return dt, msg, r.get("usage", {})


def coherence(base, model):
    print("===== chat coherence =====")
    cases = [
        ("factual", [{"role": "user", "content": "What is the capital of Australia, and roughly how many people live there? Answer in two sentences."}], "canberra"),
        ("reasoning", [{"role": "user", "content": "A train leaves at 09:40 and arrives at 13:05. How long is the trip? Show the arithmetic briefly."}], "3 hours 25"),
        ("code", [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively. Only the code."}], "def "),
        ("multiturn", [
            {"role": "user", "content": "My name is Ortensia and my favourite number is 47."},
            {"role": "assistant", "content": "Nice to meet you, Ortensia! 47 is a fine prime."},
            {"role": "user", "content": "What is my favourite number plus 5, and what is my name?"}], "52"),
    ]
    ok_all = True
    for name, msgs, expect in cases:
        try:
            dt, msg, usage = chat(base, model, msgs)
            text = (msg.get("content") or "")
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            s = salad_score(text)
            hit = expect.lower() in text.lower() or expect.lower() in reasoning.lower()
            ok = hit and s["unique_ratio"] > 0.2 and s["printable"] > 0.98 and s["len"] > 0
            ok_all &= ok
            print(f"[{name}] {'OK ' if ok else 'BAD'} {dt:.1f}s {usage.get('completion_tokens')} tok "
                  f"salad={s} expect={expect!r}\n    {text[:300]!r}")
        except Exception as e:  # noqa: BLE001
            ok_all = False
            print(f"[{name}] ERROR {str(e)[:200]}")
    # tool call
    try:
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        dt, msg, usage = chat(base, model, [{"role": "user", "content": "What's the weather in Lisbon right now? Use the tool."}],
                              tools=tools, tool_choice="auto")
        calls = msg.get("tool_calls") or []
        ok = bool(calls) and "lisbon" in json.dumps(calls).lower()
        ok_all &= ok
        print(f"[toolcall] {'OK ' if ok else 'BAD'} {dt:.1f}s calls={json.dumps(calls)[:200]}")
    except Exception as e:  # noqa: BLE001
        ok_all = False
        print(f"[toolcall] ERROR {str(e)[:200]}")
    print("coherence:", "PASS" if ok_all else "FAIL")
    return ok_all


def needle_prompt(approx_tokens, seed, depth, passphrase):
    rng = random.Random(seed)
    n_words = int(approx_tokens / 1.3)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    at = int(n_words * depth)
    needle = f". IMPORTANT FACT: the secret passphrase is {passphrase} . "
    body = " ".join(words[:at]) + needle + " ".join(words[at:])
    return ("Read the following notes carefully.\n\n" + body +
            "\n\nQuestion: What is the secret passphrase mentioned in the notes?\n"
            "Answer: The secret passphrase is")


def needles(base, model, ctxs, conc):
    print("===== needles (depths 10/50/90 %, one request each) =====")
    print(f"{'ctx':>8} {'depth':>5} {'prompt tok':>10} {'s':>8} result")
    results = {}
    for ctx in ctxs:
        for depth in (0.1, 0.5, 0.9):
            pw = f"amber-{random.Random(ctx * 7 + int(depth * 10)).randrange(1000, 9999)}-lantern"
            try:
                dt, r = post(f"{base}/v1/completions",
                             {"model": model, "prompt": needle_prompt(ctx, ctx + int(depth * 100), depth, pw),
                              "max_tokens": 24, "temperature": 0})
                text = r["choices"][0]["text"]
                ptok = r["usage"]["prompt_tokens"]
                ok = pw in text
                results[(ctx, depth)] = ok
                print(f"{ctx:>8} {depth:>5} {ptok:>10} {dt:>8.1f} {'PASS' if ok else 'WRONG'} {text.strip()[:60]!r}")
            except Exception as e:  # noqa: BLE001
                results[(ctx, depth)] = False
                print(f"{ctx:>8} {depth:>5} {'-':>10} {'-':>8} DEAD/ERR {str(e)[:120]}")
            sys.stdout.flush()
    if conc > 1:
        ctx = ctxs[min(len(ctxs) - 1, 1)]
        print(f"===== {conc} concurrent needles at ~{ctx} tokens, distinct passphrases =====")
        pws = [f"cobalt-{1000 + i * 137}-meadow" for i in range(conc)]
        outs = [None] * conc

        def work(i):
            try:
                _, r = post(f"{base}/v1/completions",
                            {"model": model, "prompt": needle_prompt(ctx, 5000 + i, 0.3, pws[i]),
                             "max_tokens": 24, "temperature": 0})
                outs[i] = r["choices"][0]["text"]
            except Exception as e:  # noqa: BLE001
                outs[i] = f"ERR {e}"
        ths = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
        t0 = time.perf_counter()
        [t.start() for t in ths]
        [t.join() for t in ths]
        for i, o in enumerate(outs):
            own = pws[i] in (o or "")
            other = any(p in (o or "") for j, p in enumerate(pws) if j != i)
            results[("conc", i)] = own and not other
            print(f"  req {i}: {'PASS' if own and not other else 'FAIL'} {'(BLEED!)' if other else ''} {str(o).strip()[:50]!r}")
        print(f"  wall {time.perf_counter() - t0:.1f}s")
    n_ok = sum(results.values())
    print(f"needles: {n_ok}/{len(results)} passed")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--ctx", default="4000,32000,128000,512000,1000000")
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--skip-chat", action="store_true")
    ap.add_argument("--skip-needles", action="store_true")
    args = ap.parse_args()
    if not args.skip_chat:
        coherence(args.url, args.model)
    if not args.skip_needles:
        needles(args.url, args.model, [int(x) for x in args.ctx.split(",")], args.conc)


if __name__ == "__main__":
    main()
