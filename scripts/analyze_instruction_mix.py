
#!/usr/bin/env python3
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "experiments/B0_measurement/resnet18_b0_pc_histogram.txt"
DISASM = ROOT / "experiments/B0_measurement/resnet18_b0_disassembly.txt"
OUT = ROOT / "experiments/B0_measurement/instruction_mix_summary.txt"


# ------------------------------------------------------------
# 1. Read PC histogram: <PC> <dynamic count>
# ------------------------------------------------------------
hist = {}

with HIST.open() as f:
    for line in f:
        m = re.match(r"^\s*([0-9a-fA-F]+)\s+(\d+)\s*$", line)
        if m:
            pc = int(m.group(1), 16)
            count = int(m.group(2))
            hist[pc] = count


# ------------------------------------------------------------
# 2. Read disassembly:
#    address: machine-code mnemonic operands
# ------------------------------------------------------------
instructions = {}

pat = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+"
    r"(?:[0-9a-fA-F]{4,8})\s+"
    r"([a-zA-Z][a-zA-Z0-9_.]*)\s*(.*?)\s*$"
)

with DISASM.open(errors="replace") as f:
    for line in f:
        m = pat.match(line)
        if m:
            pc = int(m.group(1), 16)
            mnemonic = m.group(2).lower()
            operands = m.group(3)
            instructions[pc] = (mnemonic, operands)


# ------------------------------------------------------------
# 3. Instruction classification
#
# Required top-level categories:
#   load, store, integer_alu, multiply, branch
#
# Integer ALU subcategories:
#   add, non_add, divide_remainder
#
# "branch" includes conditional branches and control transfers
# (jumps/calls/returns), matching the existing report convention.
# ------------------------------------------------------------

LOADS = {
    "lb", "lbu", "lh", "lhu", "lw", "lwu", "ld",
    "c.lb", "c.lbu", "c.lh", "c.lhu", "c.lw", "c.ld",
    "c.lwsp", "c.ldsp",
}

STORES = {
    "sb", "sh", "sw", "sd",
    "c.sb", "c.sh", "c.sw", "c.sd",
    "c.swsp", "c.sdsp",
}

CONDITIONAL_BRANCHES = {
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "beqz", "bnez", "bltz", "bgez", "blez", "bgtz",
    "c.beqz", "c.bnez",
}

CONTROL_TRANSFERS = {
    "j", "jal", "jalr", "jr", "ret",
    "call", "tail",
    "c.j", "c.jal", "c.jr", "c.jalr",
}

# Add-family arithmetic instructions.
# Aliases such as mv/li are not counted as add operations here:
# they are classified as non-add integer ALU instructions.
ADD_PREFIXES = ("add", "c.add")

# Integer ALU operations other than addition.
NON_ADD_PREFIXES = (
    "sub", "c.sub",
    "and", "c.and",
    "or", "c.or",
    "xor", "c.xor",
    "sll", "c.sll",
    "srl", "c.srl",
    "sra", "c.sra",
    "slt", "c.slt",
    "seq", "sne", "sge", "sgt",
    "slti", "sltiu",
    "andi", "ori", "xori",
    "slli", "srli", "srai",
    "lui", "auipc",
    "neg", "not",
    "sext", "zext",
)

NON_ADD_EXACT = {
    "mv", "li", "nop",
    "negw", "sext.w", "zext.w",
    "seqz", "snez", "sltz", "sgtz",
    "blez", "bgez", "bltz", "bgtz",
}


def classify(mn):
    """Return (top_level_category, subcategory)."""

    # Load/store
    if mn in LOADS:
        return "load", "load"

    if mn in STORES:
        return "store", "store"

    # Multiply family, including mulw and high-half multiply variants
    if mn.startswith("mul"):
        return "multiply", "multiply"

    # Divide/remainder are counted separately in the ALU breakdown,
    # but remain part of the required top-level integer_alu category.
    if mn.startswith(("div", "rem")):
        return "integer_alu", "divide_remainder"

    # Conditional branches and jumps/calls/returns
    if mn in CONDITIONAL_BRANCHES or mn in CONTROL_TRANSFERS:
        return "branch", "branch"

    # Addition family
    if mn.startswith(ADD_PREFIXES):
        return "integer_alu", "add"

    # Other integer ALU operations
    if mn in NON_ADD_EXACT or mn.startswith(NON_ADD_PREFIXES):
        return "integer_alu", "non_add"

    return "other", "other"


# ------------------------------------------------------------
# 4. Accumulate dynamic counts
# ------------------------------------------------------------
top_counts = Counter()
sub_counts = Counter()
mnemonic_counts = Counter()
missing_pcs = []
missing_dynamic = 0
unclassified_mnemonics = Counter()

for pc, dynamic_count in hist.items():
    inst = instructions.get(pc)

    if inst is None:
        missing_pcs.append(pc)
        missing_dynamic += dynamic_count
        continue

    mnemonic, _ = inst
    category, subcategory = classify(mnemonic)

    top_counts[category] += dynamic_count
    sub_counts[subcategory] += dynamic_count
    mnemonic_counts[mnemonic] += dynamic_count

    if category == "other":
        unclassified_mnemonics[mnemonic] += dynamic_count


hist_total = sum(hist.values())
matched_total = sum(top_counts.values())
classified_total = (
    top_counts["load"]
    + top_counts["store"]
    + top_counts["integer_alu"]
    + top_counts["multiply"]
    + top_counts["branch"]
)
unclassified_total = top_counts["other"]

# The subcategories should sum to the top-level integer ALU count.
alu_subtotal = (
    sub_counts["add"]
    + sub_counts["non_add"]
    + sub_counts["divide_remainder"]
)


def percentage(n, denominator):
    return 100.0 * n / denominator if denominator else 0.0


# ------------------------------------------------------------
# 5. Generate report
# ------------------------------------------------------------
report = []

report.append("B0 Dynamic Instruction Mix")
report.append("=" * 72)
report.append(f"Histogram file: {HIST.relative_to(ROOT)}")
report.append(f"Disassembly file: {DISASM.relative_to(ROOT)}")
report.append("")
report.append(f"Histogram PCs: {len(hist):,}")
report.append(f"Histogram dynamic-count sum: {hist_total:,}")
report.append(f"Matched dynamic-count sum: {matched_total:,}")
report.append(f"PCs without disassembly match: {len(missing_pcs):,}")
report.append(f"Dynamic count without disassembly match: {missing_dynamic:,}")
report.append("")

report.append("Top-level instruction mix")
report.append("-" * 72)
report.append(
    f"{'Category':<20}{'Dynamic count':>22}{'Share of matched':>22}"
)

for category in (
    "load",
    "store",
    "integer_alu",
    "multiply",
    "branch",
    "other",
):
    n = top_counts[category]
    report.append(
        f"{category:<20}{n:>22,}{percentage(n, matched_total):>21.3f}%"
    )

report.append("")
report.append("Integer ALU subcategories")
report.append("-" * 72)
report.append(
    f"{'Subcategory':<20}{'Dynamic count':>22}{'Share of integer ALU':>26}"
)

for category in ("add", "non_add", "divide_remainder"):
    n = sub_counts[category]
    report.append(
        f"{category:<20}{n:>22,}"
        f"{percentage(n, top_counts['integer_alu']):>25.3f}%"
    )

report.append("")
report.append("Derived metrics")
report.append("-" * 72)
report.append(f"Dynamic memory instructions (load + store): "
              f"{top_counts['load'] + top_counts['store']:,}")
report.append(f"Dynamic control-transfer instructions: "
              f"{top_counts['branch']:,}")
report.append(f"Dynamic multiply instructions: "
              f"{top_counts['multiply']:,}")
report.append(f"Dynamic add-family instructions: "
              f"{sub_counts['add']:,}")
report.append(f"Dynamic non-add ALU instructions: "
              f"{sub_counts['non_add']:,}")
report.append(f"Dynamic divide/remainder instructions: "
              f"{sub_counts['divide_remainder']:,}")

report.append("")
report.append("Top 30 mnemonics by dynamic count")
report.append("-" * 72)
report.append(
    f"{'Mnemonic':<16}{'Dynamic count':>22}{'Share of matched':>22}"
)

for mnemonic, n in mnemonic_counts.most_common(30):
    report.append(
        f"{mnemonic:<16}{n:>22,}{percentage(n, matched_total):>21.3f}%"
    )

report.append("")
report.append("Unclassified mnemonics by dynamic count")
report.append("-" * 72)

if unclassified_mnemonics:
    for mnemonic, n in unclassified_mnemonics.most_common(30):
        report.append(f"{mnemonic:<16}{n:>22,}")
else:
    report.append("None")

report.append("")
report.append("Validation")
report.append("-" * 72)
report.append(f"Matched + unmatched dynamic count: "
              f"{matched_total + missing_dynamic:,}")
report.append(f"Histogram total: {hist_total:,}")
report.append(f"Top-level classified total (excluding other): "
              f"{classified_total:,}")
report.append(f"Integer ALU top-level count: "
              f"{top_counts['integer_alu']:,}")
report.append(f"Integer ALU subcategory sum: {alu_subtotal:,}")
report.append(
    "ALU subcategories match top-level ALU: "
    + ("YES" if alu_subtotal == top_counts["integer_alu"] else "NO")
)
report.append(
    "Histogram accounting matches: "
    + ("YES" if matched_total + missing_dynamic == hist_total else "NO")
)
report.append("")
report.append("Notes:")
report.append("- Five top-level categories are retained for project reporting.")
report.append("- Integer ALU is subdivided into add, non_add, and divide_remainder.")
report.append("- Branch includes conditional branches and control transfers.")
report.append("- Unmatched PCs and 'other' instructions require review.")
report.append("- Instruction counts are not cycle counts.")

OUT.write_text("\n".join(report) + "\n")

print("\n".join(report))
print(f"\nSaved: {OUT}")



