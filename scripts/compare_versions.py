
#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from collections import defaultdict


# ==================================================
# Constants and Regex
# ==================================================

EXPECTED_OUTPUT_ELEMENTS = 1000

HISTOGRAM_RE = re.compile(
    r"^\s*(?:0x)?([0-9a-fA-F]+)\s+(\d+)\s*$"
)

HISTOGRAM_SIZE_RE = re.compile(
    r"PC Histogram size:\s*(\d+)"
)

# Supports compressed and normal RISC-V instructions.
# Also supports objdump's ".insn" pseudo mnemonic.
DISASM_RE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+"
    r"((?:(?:[0-9a-fA-F]{4}|[0-9a-fA-F]{8})\s+)+)"
    r"([.a-zA-Z][a-zA-Z0-9._]*)\b"
    r"(.*)$"
)

OUTPUT_RE = re.compile(
    r"^\s*OUTPUT_INT8\s+(\d+)\s+(-?\d+)\s*$"
)


# ==================================================
# Parsing
# ==================================================

def parse_histogram(path):
    histogram = defaultdict(int)
    found_marker = False
    declared_size = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            if "PC Histogram size:" in line:
                found_marker = True

                match = HISTOGRAM_SIZE_RE.search(line)
                if match:
                    declared_size = int(match.group(1))

                continue

            if not found_marker:
                continue

            match = HISTOGRAM_RE.match(line)

            if match:
                pc = int(match.group(1), 16)
                count = int(match.group(2))
                histogram[pc] += count

    if not found_marker:
        raise ValueError(
            f"Cannot find 'PC Histogram size:' in {path}"
        )

    if not histogram:
        raise ValueError(
            f"No histogram entries found in {path}"
        )

    return dict(histogram), declared_size


def parse_disassembly(path, custom_name, custom_match, custom_mask):
    instructions = {}

    with open(path, "r", errors="replace") as f:
        for line in f:
            match = DISASM_RE.match(line)

            if not match:
                continue

            pc = int(match.group(1), 16)
            machine_words = match.group(2).split()
            mnemonic = match.group(3)
            operands = match.group(4).strip()

            assembly = f"{mnemonic} {operands}".strip()

            # Decode a custom instruction from its 32-bit encoding.
            # objdump may display it as ".insn" instead of its real name.
            if custom_name:
                for word in machine_words:
                    if len(word) == 8:
                        encoding = int(word, 16)

                        if (encoding & custom_mask) == custom_match:
                            mnemonic = custom_name
                            assembly = (
                                f"{custom_name} "
                                f"(encoding=0x{encoding:08x})"
                            )
                            break

            instructions[pc] = (mnemonic, assembly)

    return instructions


def parse_output_scores(path):
    scores = {}
    duplicate_indices = []
    found_histogram_marker = False

    with open(path, "r", errors="replace") as f:
        for line in f:
            if "PC Histogram size:" in line:
                found_histogram_marker = True
                break

            match = OUTPUT_RE.match(line)

            if not match:
                continue

            index = int(match.group(1))
            value = int(match.group(2))

            if index in scores:
                duplicate_indices.append(index)

            scores[index] = value

    return scores, duplicate_indices


# ==================================================
# Analysis
# ==================================================

def analyze(histogram, disassembly, declared_size):
    total = sum(histogram.values())

    mapped_count = 0
    unmapped_count = 0
    mnemonic_counts = defaultdict(int)

    for pc, count in histogram.items():
        if pc in disassembly:
            mnemonic, _ = disassembly[pc]
            mapped_count += count
            mnemonic_counts[mnemonic] += count
        else:
            unmapped_count += count

    top50 = sorted(
        histogram.items(),
        key=lambda x: (-x[1], x[0])
    )[:50]

    return {
        "total": total,
        "unique_pcs": len(histogram),
        "declared_size": declared_size,
        "mapped_count": mapped_count,
        "unmapped_count": unmapped_count,
        "mapped_percentage": (
            mapped_count / total * 100 if total else 0
        ),
        "mnemonic_counts": dict(mnemonic_counts),
        "top50": top50,
        "disassembly": disassembly,
    }


def analyze_output_scores(scores, duplicate_indices, expected_elements):
    expected_indices = set(range(expected_elements))
    actual_indices = set(scores.keys())

    missing_indices = sorted(expected_indices - actual_indices)
    unexpected_indices = sorted(actual_indices - expected_indices)

    return {
        "count": len(scores),
        "missing_indices": missing_indices,
        "unexpected_indices": unexpected_indices,
        "duplicate_indices": sorted(set(duplicate_indices)),
        "complete": (
            not missing_indices
            and not unexpected_indices
            and not duplicate_indices
        ),
    }


# ==================================================
# Formatting
# ==================================================

def percentage_change(a, b):
    if a == 0:
        return "N/A"

    return f"{(b - a) / a * 100:+.4f}%"


def write_output_status(f, scores, info, expected_elements):
    f.write("[Output Tensor]\n")
    f.write("-" * 75 + "\n")

    f.write(f"Parsed output elements   : {info['count']}\n")
    f.write(f"Expected output elements : {expected_elements}\n")
    f.write(
        f"Output complete          : "
        f"{'YES' if info['complete'] else 'NO'}\n"
    )

    if info["missing_indices"]:
        f.write(f"Missing indices          : {info['missing_indices']}\n")

    if info["unexpected_indices"]:
        f.write(
            f"Unexpected indices       : "
            f"{info['unexpected_indices']}\n"
        )

    if info["duplicate_indices"]:
        f.write(
            f"Duplicate indices        : "
            f"{info['duplicate_indices']}\n"
        )

    if info["complete"]:
        top1_index = max(scores, key=scores.get)
        f.write(f"Top-1 index               : {top1_index}\n")
        f.write(
            f"Top-1 quantized score     : "
            f"{scores[top1_index]}\n"
        )

    f.write("\n")


# ==================================================
# Individual Summary
# ==================================================

def write_summary(result, scores, output_info,
                  output_path, name, expected_elements):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 75 + "\n")
        f.write(f"{name} INSTRUCTION HISTOGRAM SUMMARY\n")
        f.write("=" * 75 + "\n\n")

        # Overall statistics
        f.write("[1] Overall Statistics\n")
        f.write("-" * 75 + "\n")

        f.write(f"Dynamic instruction count : {result['total']:,}\n")
        f.write(f"Unique PCs                : {result['unique_pcs']:,}\n")

        declared = result["declared_size"]
        f.write(
            f"Declared histogram size   : "
            f"{declared:,}\n" if declared is not None
            else "Declared histogram size   : N/A\n"
        )

        f.write(f"Mapped execution count    : {result['mapped_count']:,}\n")
        f.write(f"Unmapped execution count  : {result['unmapped_count']:,}\n")
        f.write(
            f"Mapped percentage        : "
            f"{result['mapped_percentage']:.4f}%\n\n"
        )

        # Top 50 PCs
        f.write("[2] Top 50 PCs\n")
        f.write("-" * 75 + "\n")
        f.write(
            f"{'Rank':<6}{'PC':<20}{'Count':>18}"
            f"{'Percentage':>14}  Assembly\n"
        )
        f.write("-" * 130 + "\n")

        for rank, (pc, count) in enumerate(result["top50"], 1):
            percentage = (
                count / result["total"] * 100
                if result["total"] else 0
            )

            if pc in result["disassembly"]:
                _, assembly = result["disassembly"][pc]
            else:
                assembly = "<UNMAPPED>"

            f.write(
                f"{rank:<6}0x{pc:016x}  "
                f"{count:>18,}{percentage:>13.4f}%"
                f"  {assembly}\n"
            )

        # Instruction mix
        f.write("\n[3] Dynamic Instruction Mix (Mapped PCs)\n")
        f.write("-" * 75 + "\n")
        f.write(
            f"{'Mnemonic':<18}{'Dynamic Count':>20}"
            f"{'Percentage':>15}\n"
        )
        f.write("-" * 55 + "\n")

        sorted_mix = sorted(
            result["mnemonic_counts"].items(),
            key=lambda x: -x[1]
        )

        for mnemonic, count in sorted_mix:
            percentage = (
                count / result["mapped_count"] * 100
                if result["mapped_count"] else 0
            )

            f.write(
                f"{mnemonic:<18}{count:>20,}"
                f"{percentage:>14.4f}%\n"
            )

        f.write("\n")
        write_output_status(
            f, scores, output_info, expected_elements
        )


# ==================================================
# Comparison
# ==================================================

def write_comparison(a_name, b_name, a, b,
                     scores_a, scores_b,
                     info_a, info_b,
                     output_path, expected_elements):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 85 + "\n")
        f.write(f"{a_name} vs {b_name} COMPARISON\n")
        f.write("=" * 85 + "\n\n")

        # Overall statistics
        f.write("[1] Overall Statistics\n")
        f.write("-" * 85 + "\n")

        metrics = [
            ("Dynamic instruction count", "total"),
            ("Unique PCs", "unique_pcs"),
            ("Mapped execution count", "mapped_count"),
            ("Unmapped execution count", "unmapped_count"),
        ]

        f.write(
            f"{'Metric':<32}{a_name:>20}"
            f"{b_name:>20}{'Change':>15}\n"
        )
        f.write("-" * 90 + "\n")

        for label, key in metrics:
            va = a[key]
            vb = b[key]

            f.write(
                f"{label:<32}{va:>20,}{vb:>20,}"
                f"{percentage_change(va, vb):>15}\n"
            )

        f.write(
            f"\n{'Mapped percentage':<32}"
            f"{a['mapped_percentage']:>19.4f}%"
            f"{b['mapped_percentage']:>19.4f}%\n\n"
        )

        # Instruction mix
        f.write("[2] Dynamic Instruction Mix Comparison\n")
        f.write("-" * 85 + "\n")

        mix_a = a["mnemonic_counts"]
        mix_b = b["mnemonic_counts"]

        all_mnemonics = sorted(
            set(mix_a) | set(mix_b),
            key=lambda m: -(mix_a.get(m, 0) + mix_b.get(m, 0))
        )

        f.write(
            f"{'Mnemonic':<16}{a_name + ' Count':>18}"
            f"{b_name + ' Count':>18}"
            f"{a_name + ' %':>12}{b_name + ' %':>12}"
            f"{'Count Change':>16}\n"
        )
        f.write("-" * 95 + "\n")

        for mnemonic in all_mnemonics:
            ca = mix_a.get(mnemonic, 0)
            cb = mix_b.get(mnemonic, 0)

            pa = ca / a["mapped_count"] * 100 if a["mapped_count"] else 0
            pb = cb / b["mapped_count"] * 100 if b["mapped_count"] else 0

            f.write(
                f"{mnemonic:<16}{ca:>18,}{cb:>18,}"
                f"{pa:>11.4f}%{pb:>11.4f}%"
                f"{percentage_change(ca, cb):>16}\n"
            )

        # Top PCs
        f.write("\n[3] Top 50 PCs (Independent Lists)\n")
        f.write("-" * 85 + "\n")
        f.write(
            "Note: Different binaries may have different PC addresses. "
            "PCs are not compared.\n\n"
        )

        for name, result in [(a_name, a), (b_name, b)]:
            f.write(f"{name} Top 50 PCs\n")
            f.write("-" * 75 + "\n")
            f.write(
                f"{'Rank':<6}{'PC':<20}{'Count':>18}"
                f"{'Percentage':>14}  Assembly\n"
            )
            f.write("-" * 130 + "\n")

            for rank, (pc, count) in enumerate(result["top50"], 1):
                percentage = (
                    count / result["total"] * 100
                    if result["total"] else 0
                )

                if pc in result["disassembly"]:
                    _, assembly = result["disassembly"][pc]
                else:
                    assembly = "<UNMAPPED>"

                f.write(
                    f"{rank:<6}0x{pc:016x}  "
                    f"{count:>18,}{percentage:>13.4f}%"
                    f"  {assembly}\n"
                )

            f.write("\n")

        # Output tensor comparison
        f.write("[4] INT8 Output Tensor Comparison\n")
        f.write("-" * 85 + "\n")

        f.write(f"{a_name} parsed elements : {info_a['count']}\n")
        f.write(f"{b_name} parsed elements : {info_b['count']}\n")
        f.write(
            f"{a_name} complete        : "
            f"{'YES' if info_a['complete'] else 'NO'}\n"
        )
        f.write(
            f"{b_name} complete        : "
            f"{'YES' if info_b['complete'] else 'NO'}\n\n"
        )

        if info_a["complete"] and info_b["complete"]:
            differences = [
                (i, scores_a[i], scores_b[i])
                for i in range(expected_elements)
                if scores_a[i] != scores_b[i]
            ]

            f.write(
                f"Output tensors identical : "
                f"{'YES' if not differences else 'NO'}\n"
            )
            f.write(f"Different elements       : {len(differences)}\n")
            f.write(
                f"Identical elements       : "
                f"{expected_elements - len(differences)}\n"
            )

            if differences:
                f.write("\nDifferent output elements:\n")
                f.write(
                    f"{'Index':<12}{a_name + ' Value':>18}"
                    f"{b_name + ' Value':>18}{'Delta':>15}\n"
                )
                f.write("-" * 63 + "\n")

                for index, value_a, value_b in differences:
                    f.write(
                        f"{index:<12}{value_a:>18}"
                        f"{value_b:>18}{value_b - value_a:>15}\n"
                    )
        else:
            f.write(
                "Output tensors identical : "
                "NOT CHECKED (incomplete or invalid output)\n"
            )


# ==================================================
# Main
# ==================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare instruction histograms and outputs "
                    "of any two versions."
    )

    parser.add_argument("--a-name", required=True)
    parser.add_argument("--a-hist", required=True)
    parser.add_argument("--a-dis", required=True)
    parser.add_argument("--a-summary", required=True)

    parser.add_argument("--b-name", required=True)
    parser.add_argument("--b-hist", required=True)
    parser.add_argument("--b-dis", required=True)
    parser.add_argument("--b-summary", required=True)

    parser.add_argument("--comparison", required=True)

    parser.add_argument(
        "--expected-elements",
        type=int,
        default=EXPECTED_OUTPUT_ELEMENTS
    )

    parser.add_argument(
        "--custom-name",
        default="xqmac8",
        help="Name used to identify a custom instruction in disassembly."
    )
    parser.add_argument(
        "--custom-match",
        type=lambda x: int(x, 0),
        default=0x0000000B
    )
    parser.add_argument(
        "--custom-mask",
        type=lambda x: int(x, 0),
        default=0xFE00707F
    )
    parser.add_argument(
        "--no-custom-decode",
        action="store_true",
        help="Disable custom instruction encoding recognition."
    )

    args = parser.parse_args()

    custom_name = None if args.no_custom_decode else args.custom_name

    # Analyze A
    print(f"[1/4] Parsing {args.a_name}...")

    hist_a, size_a = parse_histogram(args.a_hist)
    dis_a = parse_disassembly(
        args.a_dis, custom_name,
        args.custom_match, args.custom_mask
    )
    result_a = analyze(hist_a, dis_a, size_a)

    scores_a, duplicates_a = parse_output_scores(args.a_hist)
    info_a = analyze_output_scores(
        scores_a, duplicates_a, args.expected_elements
    )

    write_summary(
        result_a, scores_a, info_a,
        args.a_summary, args.a_name, args.expected_elements
    )

    # Analyze B
    print(f"[2/4] Parsing {args.b_name}...")

    hist_b, size_b = parse_histogram(args.b_hist)
    dis_b = parse_disassembly(
        args.b_dis, custom_name,
        args.custom_match, args.custom_mask
    )
    result_b = analyze(hist_b, dis_b, size_b)

    scores_b, duplicates_b = parse_output_scores(args.b_hist)
    info_b = analyze_output_scores(
        scores_b, duplicates_b, args.expected_elements
    )

    write_summary(
        result_b, scores_b, info_b,
        args.b_summary, args.b_name, args.expected_elements
    )

    # Compare
    print(f"[3/4] Comparing {args.a_name} and {args.b_name}...")

    write_comparison(
        args.a_name, args.b_name,
        result_a, result_b,
        scores_a, scores_b,
        info_a, info_b,
        args.comparison, args.expected_elements
    )

    # Result
    print("[4/4] Done!\n")

    print(
        f"{args.a_name} dynamic instructions: "
        f"{result_a['total']:,}"
    )
    print(
        f"{args.b_name} dynamic instructions: "
        f"{result_b['total']:,}"
    )
    print(
        "Dynamic instruction change: "
        f"{percentage_change(result_a['total'], result_b['total'])}"
    )

    print(
        f"{args.a_name} output elements: {info_a['count']}"
    )
    print(
        f"{args.b_name} output elements: {info_b['count']}"
    )

    if info_a["complete"] and info_b["complete"]:
        differences = sum(
            scores_a[i] != scores_b[i]
            for i in range(args.expected_elements)
        )
        print(f"Different output elements: {differences}")
        print(
            "Output tensors identical: "
            f"{'YES' if differences == 0 else 'NO'}"
        )
    else:
        print("Output tensors identical: NOT CHECKED")

    print("\nGenerated files:")
    print(f"  {args.a_summary}")
    print(f"  {args.b_summary}")
    print(f"  {args.comparison}")


if __name__ == "__main__":
    main()



# python3 compare_versions.py \
#     --a-name B1 \
#     --a-hist B1/results/b1_histogram_output.txt \
#     --a-dis B1/results/b1.dis \
#     --a-summary B1/results/b1_summary.txt \
#     --b-name X1-Xqmac8 \
#     --b-hist X1/xqmac8/results/x1_histogram_output.txt \
#     --b-dis X1/xqmac8/results/x1_xqmac8.dis \
#     --b-summary X1/xqmac8/results/x1_xqmac8_summary.txt \
#     --comparison B1_vs_X1_xqmac8_comparison.txt

