from pathlib import Path
from collections import Counter
import csv
import re


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results")
INPUT = ROOT / "resnet18_b0_pc_histogram.txt"
OUT = ROOT / "histogram_analysis"
OUT.mkdir(parents=True, exist_ok=True)


# Parse rows such as:
# 800049c6 1
# ffffffc0000022dc 12345
pattern = re.compile(r"^\s*(0x[0-9a-fA-F]+|[0-9a-fA-F]+)\s+(\d+)\s*$")


entries = []
bad_lines = []


for line_no, line in enumerate(
    INPUT.read_text(errors="replace").splitlines(), start=1
):
    if "PC Histogram size:" in line:
        continue


    match = pattern.match(line)
    if not match:
        if line.strip():
            bad_lines.append((line_no, line))
        continue


    pc = int(match.group(1), 16)
    count = int(match.group(2))
    entries.append((pc, count))


if not entries:
    raise SystemExit("No histogram entries parsed. Check the input format.")


# Consolidate duplicate PC rows, if any.
counts = Counter()
for pc, count in entries:
    counts[pc] += count


ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
total = sum(counts.values())
unique_pcs = len(counts)
count_values = [count for _, count in ranked]


def pct(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else 0.0


summary = [
    f"Input file: {INPUT}",
    f"Parsed rows: {len(entries):,}",
    f"Unique PCs: {unique_pcs:,}",
    f"Sum of histogram counts: {total:,}",
    f"Maximum PC count: {ranked[0][1]:,}",
    f"Minimum PC count: {min(count_values):,}",
    f"Median PC count: {sorted(count_values)[len(count_values)//2]:,}",
    "",
    "Top-PC coverage:",
]


for n in (10, 20, 50, 100, 500, 1000):
    covered = sum(count for _, count in ranked[:n])
    summary.append(
        f"Top {min(n, unique_pcs):,} PCs: "
        f"{covered:,} counts ({pct(covered, total):.3f}%)"
    )


for threshold in (1, 10, 100, 1000, 10000, 1000000):
    n = sum(1 for count in count_values if count >= threshold)
    covered = sum(count for count in count_values if count >= threshold)
    summary.append(
        f"PC count >= {threshold:,}: {n:,} PCs; "
        f"{pct(covered, total):.3f}% of summed counts"
    )


(OUT / "summary.txt").write_text("\n".join(summary) + "\n")
print("\n".join(summary))


with (OUT / "top_pcs.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "pc_hex", "pc_decimal", "count", "cumulative_pct"])
    cumulative = 0
    for rank, (pc, count) in enumerate(ranked, start=1):
        cumulative += count
        writer.writerow([
            rank, f"0x{pc:x}", pc, count, f"{pct(cumulative, total):.6f}"
        ])


# Plot 1: Top 30 PCs. Log scale helps show differences between hot PCs.
top_n = min(30, unique_pcs)
labels = [f"{pc:x}" for pc, _ in ranked[:top_n]]
values = [count for _, count in ranked[:top_n]]


plt.figure(figsize=(12, 6))
plt.bar(range(top_n), values)
plt.yscale("log")
plt.xticks(range(top_n), labels, rotation=75, ha="right")
plt.xlabel("PC address (hex)")
plt.ylabel("Execution count (log scale)")
plt.title(f"Top {top_n} PCs by execution count")
plt.tight_layout()
plt.savefig(OUT / "top30_pcs.png", dpi=160)
plt.close()


# Plot 2: Rank versus count, to show how quickly counts fall.
plt.figure(figsize=(9, 5))
plt.plot(range(1, unique_pcs + 1), count_values)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("PC rank (log scale)")
plt.ylabel("Execution count (log scale)")
plt.title("PC execution-count rank distribution")
plt.tight_layout()
plt.savefig(OUT / "pc_rank_distribution.png", dpi=160)
plt.close()


# Plot 3: Pareto curve, cumulative share of all histogram counts.
cumulative_counts = []
running = 0
for _, count in ranked:
    running += count
    cumulative_counts.append(pct(running, total))


plt.figure(figsize=(9, 5))
plt.plot(range(1, unique_pcs + 1), cumulative_counts)
plt.axhline(90, linestyle="--", label="90%")
plt.axhline(95, linestyle="--", label="95%")
plt.xscale("log")
plt.xlabel("Number of hottest PCs included (log scale)")
plt.ylabel("Cumulative share of summed counts (%)")
plt.title("PC histogram Pareto curve")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "pc_pareto.png", dpi=160)
plt.close()


if bad_lines:
    (OUT / "unparsed_lines.txt").write_text(
        "\n".join(f"{n}: {line}" for n, line in bad_lines) + "\n"
    )
    print(f"\nWarning: {len(bad_lines)} non-empty lines were not parsed.")
    print(f"See {OUT / 'unparsed_lines.txt'}")


print(f"\nGenerated files in {OUT}/")
