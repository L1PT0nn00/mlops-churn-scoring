"""Нагрузка на /predict. С --shift шлёт «сдвинутых» клиентов, чтобы увидеть дрейф в Grafana.

    python scripts/load_test.py --n 600                 обычный трафик
    python scripts/load_test.py --n 600 --shift 1.0     трафик с дрейфом
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from churn.data import RAW_FEATURES, generate  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--shift", type=float, default=0.0)
    p.add_argument("--rps", type=float, default=50.0)
    args = p.parse_args()

    df = generate(n=args.n, seed=random.randint(0, 10_000), shift=args.shift)
    ok = errors = 0
    latencies = []
    for _, row in df.iterrows():
        body = {"customer_id": row["customer_id"], **{c: row[c] for c in RAW_FEATURES}}
        for c in ("support_calls_90d", "num_products", "age"):
            body[c] = int(body[c])
        req = urllib.request.Request(
            f"{args.url}/predict", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
            ok += 1
        except (urllib.error.URLError, TimeoutError) as e:
            errors += 1
            if errors <= 3:
                print("error:", e, file=sys.stderr)
        latencies.append(time.perf_counter() - start)
        time.sleep(max(0.0, 1 / args.rps - latencies[-1]))

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1] * 1000 if latencies else 0
    print(json.dumps({"sent": len(df), "ok": ok, "errors": errors, "p95_ms": round(p95, 1), "shift": args.shift}))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
