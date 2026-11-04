#!/usr/bin/env python3
import bisect
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "experiments/B0_measurement/resnet18_b0_pc_histogram.txt"
DISASM = ROOT / "experiments/B0_measurement/resnet18_b0_disassembly.txt"
OUT = ROOT / "experiments/B0_measurement/operator_contribution.txt"

# Parse PC histogram: address count
hist = {}
with HIST.open() as f:
    for line in f:
        m = re.match(r"^\s*([0-9a-fA-F]+)\s+(\d+)\s*$", line)
        if m:
            hist[int(m.group(1), 16)] = int(m.group(2))

# Parse demangled function labels from objdump output
symbol_re = re.compile(r"^\s*([0-9a-fA-F]+)\s+<(.+)>:\s*$")
symbols = {}
with DISASM.open(errors="replace") as f:
    for line in f:
        m = symbol_re.match(line)
        if m:
            symbols[int(m.group(1), 16)] = m.group(2)

addresses = sorted(symbols)
function_counts = Counter()
unmatched_count = 0
unmatched_pcs = 0

# Attribute each histogram PC to the closest preceding function label.
for pc, count in hist.items():
    i = bisect.bisect_right(addresses, pc) - 1
    if i < 0:
        unmatched_count += count
        unmatched_pcs += 1
        continue

    start = addresses[i]
    name = symbols[start]

    # Avoid assigning addresses outside the ELF's program code region
    # to the last symbol simply because it precedes the PC.
    if not (0x10120 <= pc < 0x10120 + 0x2be0a):
        unmatched_count += count
        unmatched_pcs += 1
        continue

    function_counts[name] += count

total_hist = sum(hist.values())
matched = sum(function_counts.values())

# Initial name-based grouping. This is a first-pass estimate, not
# a guarantee that all inlined operator work is included.
operator_patterns = [
    ("Depthwise convolution", ("DepthwiseConv", "depthwise_conv")),
    ("Convolution", ("ConvPerChannel", "Conv2D", "Conv(", "Conv::")),
    ("Fully connected", ("FullyConnected", "FullyConnectedPerChannel")),
    ("Pooling", ("AveragePool", "MaxPool", "Pool(", "Pooling")),
    ("Softmax", ("Softmax",)),
    ("Quantize / requantize", ("Quantize", "Requantize")),
]

operator_counts = Counter()
operator_functions = Counter()

for name, count in function_counts.items():
    assigned = False
    for category, patterns in operator_patterns:
        if any(p.lower() in name.lower() for p in patterns):
            operator_counts[category] += count
            operator_functions[category] += 1
            assigned = True
            break
    if not assigned:
        operator_counts["Other / unattributed"] += count

lines = [
    "B0 Function and Operator Contribution (first-pass attribution)",
    f"Histogram PCs: {len(hist):,}",
    f"Histogram dynamic-count sum: {total_hist:,}",
    f"PCs attributed to ELF code range: {len(hist) - unmatched_pcs:,}",
    f"Dynamic count attributed to ELF code range: {matched:,}",
    f"Unmatched PCs: {unmatched_pcs:,}",
    f"Unmatched dynamic count: {unmatched_count:,}",
    "",
    "Top 30 functions by dynamic count:",
]
for name, count in function_counts.most_common(30):
    pct = 100 * count / matched if matched else 0
    lines.append(f"{count:>16,}  {pct:>7.2f}%  {name}")

lines += [
    "",
    "Initial operator grouping (name-based; not yet inline-complete):",
]
for category, count in operator_counts.most_common():
    pct = 100 * count / matched if matched else 0
    nfunc = operator_functions.get(category, 0)
    lines.append(f"{count:>16,}  {pct:>7.2f}%  {nfunc:>5} functions  {category}")

OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nSaved: {OUT}")
