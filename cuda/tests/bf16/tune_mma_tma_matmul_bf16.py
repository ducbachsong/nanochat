"""tune_mma_tma_matmul_bf16.py — the timings of cuda/kernels/bf16/mma_tma_matmul_bf16.cuh found TVM's way (AutoTVM): a
cost model picks which settings the GPU runs, instead of running all ~105,000 of each product.

    pip install xgboost numpy
    nvcc ... cuda/tests/bf16/tune_mma_tma_matmul_bf16.cu -o bin/tune_mma_tma_matmul_bf16      # the sweep's build
    python cuda/tests/bf16/tune_mma_tma_matmul_bf16.py                                   # nanochat d24's 15 products
    python cuda/tests/bf16/tune_mma_tma_matmul_bf16.py --layers attention_output head    # only those layers
    python cuda/tests/bf16/tune_mma_tma_matmul_bf16.py --resume cuda/tests/results/H100/mma_tma_matmul_bf16_model_<date>.csv

The binary, started with serve, plans every setting of a product (the sweep's lists and skips) and runs the ones
asked for: each checked to the bit against mma_matmul_bf16, then timed, as the sweep does. For each product:

    1. FIRST_RANDOM settings at random (FIRST_RANDOM_TRANSFER once other products have taught the model)
    2. XGBoost fitted with a rank loss (rank:pairwise, AutoTVM's) on every setting run so far, each product a group
       of its own: it learns which of two settings is faster, not their milliseconds
    3. every setting not run yet scored; the BATCH - RANDOM_A_BATCH best run, and RANDOM_A_BATCH at random
    4. again from 2, until the product's budget is spent, or PATIENCE rounds bring nothing GAIN faster
    5. the FINALISTS fastest timed again over SWEEP_ROUNDS rounds, taking turns with the defaults and cuBLAS

A product's budget: LEAST_A_PRODUCT settings, and its share of SHARED_A_PRODUCT x products by its flops (the head's
products get the most: they take the most time). Every setting run goes to the .csv at once: --resume picks up from
it (a spot machine gone), skipping what was run and the products already finished (their rows in _final.csv). A
setting that breaks the GPU while running is kept as failed, never asked again, and the binary started again.

Files, in cuda/tests/results/<GPU>/: mma_tma_matmul_bf16_model_<date>.txt (this console and the binary's, with the
kernels' table), .csv (every setting run), _final.csv (each product's finalists, timed again).
"""
import argparse
import csv
import datetime
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIRST_RANDOM = 256              # settings at random before the model knows anything
FIRST_RANDOM_TRANSFER = 64      # at random, once other products' settings have taught it
BATCH = 64                      # settings a round
RANDOM_A_BATCH = 4              # of them at random (AutoTVM's epsilon-greedy, ~5%): the model is not always right
LEAST_A_PRODUCT = 768           # settings every product runs at least (or all it has)
SHARED_A_PRODUCT = 768          # more a product on average, shared out by flops
PATIENCE = 5                    # rounds bringing nothing GAIN faster that end a product, past LEAST_A_PRODUCT
GAIN = 0.005                    # faster: by this much of the best so far
FINALISTS = 50                  # the .cu's FINALISTS
TOP_PRINTED = 10                # the .cu's TOP_PRINTED
MODEL_ROUNDS = 200              # XGBoost's boosting rounds
MODEL_PARAMETERS = {"objective": "rank:pairwise", "max_depth": 6, "eta": 0.2, "min_child_weight": 1, "subsample": 1.0,
                    "lambda": 1.0, "verbosity": 0}
TERA_IN_MILLISECONDS = 1e9      # flops / ms / this: TFLOPS
REGISTERS_PER_SM = 65536
SPACE_COLUMNS = ["id", "block_rows", "block_columns", "block_k", "threads", "warps_down", "warps_across", "swizzle", "parts",
                 "blocks_per_sm", "resident_per_sm", "block_order", "registers", "sums", "shared_bytes", "squares"]
CSV_COLUMNS = ["product"] + SPACE_COLUMNS + ["round", "picked", "first_ms", "first_runs", "tflops", "same", "status"]
FINAL_COLUMNS = ["product", "rank"] + SPACE_COLUMNS + ["first_ms", "final_ms", "tflops", "x_defaults", "x_cublas", "defaults_ms",
                                                         "cublas_ms"]

# ── the binary ────────────────────────────────────────────────────────────────


class Product:
    """One product as the binary describes it: its name, shape, turned flags and flops."""

    def __init__(self, fields):
        self.index, self.name = int(fields[1]), fields[2]
        self.rows, self.columns, self.sum_length = int(fields[3]), int(fields[4]), int(fields[5])
        self.left_turned, self.right_turned, self.add_onto, self.has_bias = (int(f) for f in fields[6:10])
        self.flops = float(fields[10])


class Server:
    """The tune binary started with serve: a request a line on its stdin, its replies on its stdout, its stderr to the log."""

    def __init__(self, command):
        self.command = command
        self.log = None
        self.waiting_lines = []         # the binary's stderr before the log is open
        self.selected = None            # the product selected, to select again after a restart
        self.start()

    def start(self):
        """Starts the binary and reads its GPU and products."""
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1, cwd=REPO)
        threading.Thread(target=self.copy_stderr, args=(self.process.stderr,), daemon=True).start()
        fields = self.read()
        if fields is None or fields[0] != "ready":
            raise SystemExit(f"{' '.join(self.command)} did not start: {fields}")
        self.sms, self.gpu_folder = int(fields[1]), fields[2]
        self.products = []
        while (fields := self.read()) is not None and fields[0] != "end":
            self.products.append(Product(fields))

    def copy_stderr(self, stream):
        """Copies the binary's stderr into the log, line by line, keeping it until the log is open."""
        for line in stream:
            if self.log is None:
                self.waiting_lines.append(line)
            else:
                self.log.write(line)
                self.log.flush()

    def open_log(self, log):
        """Gives the binary's stderr its log, with what it printed before."""
        self.log = log
        log.writelines(self.waiting_lines)
        log.flush()
        self.waiting_lines = []

    def read(self):
        """Gives the binary's next reply as its fields, or None if it ended."""
        line = self.process.stdout.readline()
        return line.rstrip("\n").split("\t") if line else None

    def send(self, *fields):
        """Sends one request; False if the binary ended."""
        try:
            self.process.stdin.write("\t".join(str(f) for f in fields) + "\n")
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def restart(self):
        """Starts the binary again after a setting broke the GPU, and selects the same product."""
        self.process.wait()
        self.start()
        if self.selected is not None:
            self.select(self.selected)

    def select(self, product):
        """Selects a product: gives every setting that can run (one row each, SPACE_COLUMNS), the defaults' ms and the skipped."""
        self.selected = product
        self.send("select", product.index)
        rows = []
        while (fields := self.read()) is not None and fields[0] == "space":
            rows.append([int(f) for f in fields[1:]])
        if fields is None or fields[0] != "selected":
            raise SystemExit(f"select {product.name}: {fields}")
        return np.array(rows, dtype=np.int64).reshape(-1, len(SPACE_COLUMNS)), float(fields[2]), int(fields[4])

    def run(self, setting_id):
        """Runs one setting: ("ran", ms, runs, same), ("refused", why) or ("failed", why); restarts the binary after a failure."""
        fields = self.read() if self.send("run", setting_id) else None
        if fields is None:
            fields = ["failed", str(setting_id), "the binary ended"]
        if fields[0] == "error":
            raise SystemExit(f"run {setting_id}: {fields[1]}")
        if fields[0] == "failed":
            self.restart()
            return ("failed", fields[2])
        if fields[0] == "refused":
            return ("refused", fields[2])
        return ("ran", float(fields[2]), int(fields[3]), int(fields[4]))

    def final(self, setting_ids):
        """Times settings again, taking turns with the defaults and cuBLAS: gives {id: ms}, the defaults' ms, cuBLAS's ms."""
        self.send("final", ",".join(str(i) for i in setting_ids))
        final_ms = {}
        while (fields := self.read()) is not None and fields[0] == "final":
            final_ms[int(fields[1])] = float(fields[2])
        if fields is None or fields[0] != "final_end":
            raise SystemExit(f"final: {fields}")
        return final_ms, float(fields[1]), float(fields[2])

    def close(self):
        """Ends the binary."""
        self.send("quit")
        self.process.wait()


# ── the model ─────────────────────────────────────────────────────────────────


def features(space, product, sms):
    """Gives each setting's numbers for the model: its own, what follows from them on this GPU, and the product's shape."""
    s = {name: space[:, k].astype(np.float64) for k, name in enumerate(SPACE_COLUMNS)}
    blocks = np.where(s["blocks_per_sm"] == 0, s["squares"], s["blocks_per_sm"] * sms)    # blocks a launch
    waves = s["squares"] / (s["resident_per_sm"] * sms)                                   # squares over the blocks at once
    last_wave = waves - np.floor(waves)
    last_wave[last_wave == 0] = 1.0
    warp_rows, warp_columns = s["block_rows"] / s["warps_down"], s["block_columns"] / s["warps_across"]
    padded = (np.ceil(product.rows / s["block_rows"]) * s["block_rows"]
              * np.ceil(product.columns / s["block_columns"]) * s["block_columns"]) / (product.rows * product.columns)
    columns = [
        np.log2(s["block_rows"]), np.log2(s["block_columns"]), np.log2(s["block_k"]), np.log2(s["threads"]),
        s["warps_down"], s["warps_across"], s["swizzle"], s["parts"], s["blocks_per_sm"], (s["blocks_per_sm"] == 0) * 1.0,
        np.log2(s["block_order"]), s["registers"], s["sums"], s["shared_bytes"] / 1024, s["resident_per_sm"],
        np.log2(s["squares"]), np.log2(blocks), waves, last_wave, s["squares"] / blocks,
        np.ceil(product.sum_length / s["block_k"]),                                       # steps of k
        s["block_rows"] * s["block_columns"] / (s["block_rows"] + s["block_columns"]),     # a block's reuse of a load
        warp_rows * warp_columns / (warp_rows + warp_columns),                             # a warp's
        s["resident_per_sm"] * s["threads"],
        s["resident_per_sm"] * s["threads"] * s["registers"] / REGISTERS_PER_SM,
        s["resident_per_sm"] * s["shared_bytes"] / 1024, padded,
        np.full_like(waves, np.log2(product.rows)), np.full_like(waves, np.log2(product.columns)),
        np.full_like(waves, np.log2(product.sum_length)), np.full_like(waves, product.left_turned),
        np.full_like(waves, product.right_turned), np.full_like(waves, product.add_onto),
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def train(groups):
    """Fits XGBoost's ranker on (features, ms) groups, one a product: only settings of the same product compared."""
    import xgboost as xgb
    xs, labels, sizes = [], [], []
    for x, ms in groups:
        if len(ms) >= 2:
            xs.append(x)
            labels.append(ms.min() / ms)          # its speed against the product's fastest: 1 the fastest
            sizes.append(len(ms))
    if not xs:
        return None
    data = xgb.DMatrix(np.concatenate(xs), label=np.concatenate(labels))
    data.set_group(sizes)
    return xgb.train({**MODEL_PARAMETERS, "nthread": os.cpu_count()}, data, MODEL_ROUNDS)


def predict(model, x):
    """Gives the model's score of each setting: higher, faster."""
    import xgboost as xgb
    return model.predict(xgb.DMatrix(x))


# ── the tune ──────────────────────────────────────────────────────────────────


class Tune:
    """The run: the binary, the log and the two .csv files, every setting run so far."""

    def __init__(self, arguments):
        model_arguments = (["gpt2"] if arguments.model == "gpt2" else []) + arguments.layers
        self.server = Server([os.path.join(REPO, arguments.binary), "serve", *model_arguments])
        if arguments.resume:
            stem = os.path.splitext(arguments.resume)[0]
        else:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            stem = os.path.join(REPO, "cuda", "tests", "results", self.server.gpu_folder, f"mma_tma_matmul_bf16_model_{stamp}")
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        self.log = open(stem + ".txt", "a")
        self.server.open_log(self.log)
        self.rows = self.read_csv(stem + ".csv")
        self.finals = self.read_csv(stem + "_final.csv")
        self.csv = self.open_csv(stem + ".csv", CSV_COLUMNS)
        self.final_csv = self.open_csv(stem + "_final.csv", FINAL_COLUMNS)
        self.least, self.shared = arguments.least, arguments.shared
        self.say(f"log: {stem}.txt" + (f"  (resumed: {len(self.rows)} settings run, {len(self.finals)} finalists)" if self.rows else ""))

    @staticmethod
    def read_csv(path):
        """Gives a .csv's rows as dicts; none if it is not there."""
        if not os.path.exists(path):
            return []
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def open_csv(path, columns):
        """Opens a .csv to add to, its header written if it is new."""
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        f = open(path, "a", newline="")
        writer = csv.DictWriter(f, fieldnames=columns)
        if new:
            writer.writeheader()
            f.flush()
        return f, writer

    @staticmethod
    def write(target, row):
        """Adds one row to a .csv, at once on disk."""
        f, writer = target
        writer.writerow(row)
        f.flush()

    def say(self, line=""):
        """Prints a line to the console and the log."""
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def budget(self, product):
        """Gives a product's settings to run: LEAST_A_PRODUCT and its share of the rest, by flops."""
        products = self.server.products
        total_flops = sum(p.flops for p in products)
        return int(self.least + self.shared * len(products) * product.flops / total_flops)

    def groups(self):
        """Gives every product's settings that ran so far as (features, ms), for the model."""
        by_name = {p.name: p for p in self.server.products}
        rows_by_product = {}
        for row in self.rows:
            if row["status"] == "ran" and row["product"] in by_name:
                rows_by_product.setdefault(row["product"], []).append(row)
        groups = []
        for name, rows in rows_by_product.items():
            space = np.array([[int(row[c]) for c in SPACE_COLUMNS] for row in rows], dtype=np.int64)
            groups.append((features(space, by_name[name], self.server.sms), np.array([float(row["first_ms"]) for row in rows])))
        return groups

    def tune_product(self, product):
        """Finds one product's fastest settings, round by round, then times its finalists again: gives its result."""
        space, defaults_ms, skipped = self.server.select(product)
        x = features(space, product, self.server.sms)
        index_of_id = {int(setting_id): k for k, setting_id in enumerate(space[:, 0])}
        tried = {int(row["id"]): row for row in self.rows if row["product"] == product.name}
        budget = min(self.budget(product), len(space))
        tflops = lambda ms: product.flops / ms / TERA_IN_MILLISECONDS  # noqa: E731
        self.say(f"\n{product.name}: output {{{product.rows}, {product.columns}}}, sum of {product.sum_length}: "
                 f"{product.flops / 1e12:.2f} TFLOP; the defaults {defaults_ms:.4f} ms")
        self.say(f"    {len(space)} settings can run, {skipped} skipped; budget {budget}" + (f", {len(tried)} run before" if tried else ""))

        rng = random.Random(product.name)
        round_number = max((int(row["round"]) for row in tried.values()), default=-1) + 1
        best_ms = lambda: min((float(r["first_ms"]) for r in tried.values() if r["status"] == "ran" and r["same"] == "1"),  # noqa: E731
                              default=float("inf"))
        stale = 0
        start = time.time()
        while len(tried) < budget:
            # 1. what to run: at random until the model has data, then its best and a few at random
            untried = [k for k in range(len(space)) if int(space[k, 0]) not in tried]
            room = budget - len(tried)
            other_products_ran = any(r["status"] == "ran" for r in self.rows if r["product"] != product.name)
            first = FIRST_RANDOM_TRANSFER if other_products_ran else FIRST_RANDOM
            model = None if len(tried) < first else train(self.groups())
            if model is None:
                picked = [(k, "random") for k in rng.sample(untried, min(room, max(first - len(tried), BATCH), len(untried)))]
            else:
                scores = predict(model, x[untried])
                order = [untried[j] for j in np.argsort(-scores)]
                count = min(room, BATCH)
                best = order[:max(count - RANDOM_A_BATCH, 1)]
                rest = order[len(best):]
                picked = [(k, "model") for k in best] + [(k, "random") for k in rng.sample(rest, min(count - len(best), len(rest)))]

            # 2. run them, each row on disk at once
            before_ms = best_ms()
            for k, how in picked:
                setting_id = int(space[k, 0])
                outcome = self.server.run(setting_id)
                row = {c: int(space[k, j]) for j, c in enumerate(SPACE_COLUMNS)}
                row.update(product=product.name, round=round_number, picked=how, first_ms="", first_runs="", tflops="", same="")
                if outcome[0] == "ran":
                    row.update(first_ms=f"{outcome[1]:.5f}", first_runs=outcome[2], tflops=f"{tflops(outcome[1]):.1f}", same=outcome[3],
                               status="ran")
                    if outcome[3] != 1:
                        self.say(f"    NOT mma_matmul_bf16's bits: {describe(row)}")
                else:
                    row["status"] = f"{outcome[0]}: {outcome[1]}"
                    if outcome[0] == "failed":
                        self.say(f"    failed while running (the binary started again): {describe(row)}: {outcome[1]}")
                row = {c: str(v) for c, v in row.items()}
                self.write(self.csv, row)
                self.rows.append(row)
                tried[setting_id] = row

            # 3. the round's line; PATIENCE rounds with nothing GAIN faster end the product
            now_ms = best_ms()
            stale = 0 if now_ms < before_ms * (1 - GAIN) else stale + 1
            models = sum(1 for _, how in picked if how == "model")
            self.say(f"    round {round_number:3d}: {len(picked):3d} run ({models} model, {len(picked) - models} random), "
                     f"{len(tried):5d} of {budget}, {time.time() - start:5.0f} s; best {now_ms:.4f} ms, {tflops(now_ms):.0f} TFLOPS, "
                     f"{defaults_ms / now_ms:.2f}x the defaults")
            round_number += 1
            if stale >= PATIENCE and len(tried) >= min(self.least, len(space)):
                self.say(f"    {PATIENCE} rounds with nothing {GAIN:.1%} faster: done")
                break
        return self.final_pass(product, tried, tflops)

    def final_pass(self, product, tried, tflops):
        """Times the FINALISTS fastest again with the defaults and cuBLAS, prints them, writes them to _final.csv."""
        ran = sorted((r for r in tried.values() if r["status"] == "ran" and r["same"] == "1"), key=lambda r: float(r["first_ms"]))
        finalists = ran[:FINALISTS]
        if not finalists:
            self.say("    no setting ran with mma_matmul_bf16's bits")
            return None
        final_ms, defaults_ms, cublas_ms = self.server.final([int(r["id"]) for r in finalists])
        finalists.sort(key=lambda r: final_ms[int(r["id"])])
        self.say(f"    the defaults: {defaults_ms:.4f} ms, {tflops(defaults_ms):.0f} TFLOPS; PyTorch (cuBLAS): {cublas_ms:.4f} ms, "
                 f"{tflops(cublas_ms):.0f} TFLOPS")
        self.say(f"    the {TOP_PRINTED} fastest, timed again over the .cu's SWEEP_ROUNDS rounds with the defaults and cuBLAS")
        self.say("         ms  TFLOPS  x defaults  x cuBLAS   setting")
        for rank, row in enumerate(finalists):
            ms = final_ms[int(row["id"])]
            if rank < TOP_PRINTED:
                self.say(f"    {ms:7.4f}  {tflops(ms):6.0f}     {defaults_ms / ms:5.2f}x    {cublas_ms / ms:5.2f}x   {describe(row)}")
            final_row = {c: row[c] for c in SPACE_COLUMNS}
            final_row.update(product=product.name, rank=rank + 1, first_ms=row["first_ms"], final_ms=f"{ms:.5f}", tflops=f"{tflops(ms):.1f}",
                             x_defaults=f"{defaults_ms / ms:.3f}", x_cublas=f"{cublas_ms / ms:.3f}", defaults_ms=f"{defaults_ms:.5f}",
                             cublas_ms=f"{cublas_ms:.5f}")
            final_row = {c: str(v) for c, v in final_row.items()}
            self.write(self.final_csv, final_row)
            self.finals.append(final_row)
        return self.finals[-len(finalists)]

    def run(self):
        """Tunes every product not finished yet, then prints them side by side: gives the settings without mma_matmul_bf16's bits."""
        start = time.time()
        products = self.server.products
        self.say(f"{len(products)} products, {self.server.sms} SMs")
        results = {}
        for product in products:
            done = [r for r in self.finals if r["product"] == product.name and r["rank"] == "1"]
            if done:
                self.say(f"\n{product.name}: finished before (_final.csv)")
                results[product.name] = done[0]
            else:
                results[product.name] = self.tune_product(product)
            self.say(f"    {time.time() - start:.0f} s so far")

        self.say("\nevery_product")
        self.say(f"    {'product':28s}  {'defaults':>8s}  {'cuBLAS':>8s}  {'best':>8s}  {'x defaults':>10s}  {'x cuBLAS':>8s}   best setting")
        totals = [0.0, 0.0, 0.0]
        for product in products:
            r = results[product.name]
            if r is None:
                self.say(f"    {product.name:28s}  none")
                continue
            ms = [float(r["defaults_ms"]), float(r["cublas_ms"]), float(r["final_ms"])]
            totals = [t + m for t, m in zip(totals, ms)]
            self.say(f"    {product.name:28s}  {ms[0]:8.4f}  {ms[1]:8.4f}  {ms[2]:8.4f}  {ms[0] / ms[2]:9.2f}x  {ms[1] / ms[2]:7.2f}x   {describe(r)}")
        if totals[2] > 0:
            self.say(f"    {'all, once each':28s}  {totals[0]:8.4f}  {totals[1]:8.4f}  {totals[2]:8.4f}  {totals[0] / totals[2]:9.2f}x  "
                     f"{totals[1] / totals[2]:7.2f}x")
        not_same = [r for r in self.rows if r["status"] == "ran" and r["same"] != "1"]
        run_count = sum(1 for r in self.rows if r["product"] in results)
        self.say(f"\n{run_count} settings run in all; {len(not_same)} without mma_matmul_bf16's bits; {time.time() - start:.0f} s")
        self.server.close()
        return not_same


def describe(row):
    """Gives a setting as one line, as the .cu's describe."""
    blocks = "one a square" if row["blocks_per_sm"] in (0, "0") else f"{row['blocks_per_sm']} an SM"
    swizzle = "swizzled" if row["swizzle"] in (1, "1") else "no swizzle"
    return (f"square {int(row['block_rows']):3d} x {int(row['block_columns']):3d}, k {int(row['block_k']):2d}, {int(row['threads']):4d} threads "
            f"(warps {row['warps_down']} x {int(row['warps_across']):2d}), {swizzle:10s}, {row['parts']} parts, blocks {blocks:12s}, "
            f"order {int(row['block_order']):2d}  ({int(row['resident_per_sm']):2d} on an SM)")


def main():
    parser = argparse.ArgumentParser(description="mma_tma_matmul_bf16's settings, found by a cost model (AutoTVM's way)")
    parser.add_argument("--binary", default="bin/tune_mma_tma_matmul_bf16", help="the .cu's build, from the repo's root")
    parser.add_argument("--model", default="d24", choices=["d24", "gpt2"])
    parser.add_argument("--layers", nargs="*", default=[], help="qkv attention_output mlp_up mlp_down head; none: every one")
    parser.add_argument("--resume", help="a .csv of an earlier run: carries on from it")
    parser.add_argument("--least", type=int, default=LEAST_A_PRODUCT, help="settings every product runs at least")
    parser.add_argument("--shared", type=int, default=SHARED_A_PRODUCT, help="more a product on average, shared out by flops")
    not_same = Tune(parser.parse_args()).run()
    sys.exit(1 if not_same else 0)


if __name__ == "__main__":
    main()
