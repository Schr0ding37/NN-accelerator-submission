#!/usr/bin/env python3
"""
Analyze MAC counts in a TensorFlow Lite (.tflite) model.

Default input:
  src/model/resnet18_int8_channelwise.tflite

Default outputs:
  experiments/B0_measurement/model_mac_breakdown.csv
  experiments/B0_measurement/model_mac_summary.md

Dependencies:
  pip install tflite
(or run in an environment that provides TensorFlow's schema_py_generated module)

Scope:
  Counts multiply-accumulate operations for CONV_2D, DEPTHWISE_CONV_2D,
  FULLY_CONNECTED, and BATCH_MATMUL when their tensor shapes are available.
  Other operators are listed in the report with MAC count 0 / not counted.
  This is a static model-graph estimate; it does not use Spike execution data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional


def load_tflite_schema():
    """Load generated TFLite schema bindings from either common Python package."""
    try:
        import tflite  # type: ignore
        return tflite
    except ImportError:
        pass

    try:
        from tensorflow.lite.python import schema_py_generated as tflite  # type: ignore
        return tflite
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import TFLite schema bindings. Install them with `pip install tflite`, "
            "or run this script in an environment with TensorFlow installed."
        ) from exc


def enum_name(schema: Any, enum_class_name: str, value: int) -> str:
    enum_cls = getattr(schema, enum_class_name, None)
    if enum_cls is not None:
        # Generated schema versions may expose enum members as attributes.
        for name in dir(enum_cls):
            if name.startswith("_"):
                continue
            try:
                if getattr(enum_cls, name) == value:
                    return name
            except Exception:
                continue
    return f"UNKNOWN_{value}"


def builtin_operator_name(schema: Any, code: int) -> str:
    enum_cls = getattr(schema, "BuiltinOperator", None)
    if enum_cls is not None:
        for name in dir(enum_cls):
            if name.startswith("_"):
                continue
            try:
                if getattr(enum_cls, name) == code:
                    return name
            except Exception:
                continue
    return f"BUILTIN_{code}"


def vector_to_list(vector: Any) -> list[int]:
    if vector is None:
        return []
    try:
        return [int(x) for x in vector]
    except TypeError:
        return []


def get_tensor_shape(subgraph: Any, tensor_index: int) -> list[int]:
    if tensor_index < 0 or tensor_index >= subgraph.TensorsLength():
        return []
    tensor = subgraph.Tensors(tensor_index)
    if tensor is None:
        return []
    return vector_to_list(tensor.ShapeAsNumpy())


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def operator_inputs(op: Any) -> list[int]:
    return vector_to_list(op.InputsAsNumpy())


def operator_outputs(op: Any) -> list[int]:
    return vector_to_list(op.OutputsAsNumpy())


def calculate_operator_macs(
    op_name: str,
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
) -> tuple[Optional[int], str]:
    """Return (MAC count, explanation). None means shape/layout not recognized."""
    valid_inputs = [s for s in input_shapes if s]
    valid_outputs = [s for s in output_shapes if s]
    if not valid_outputs:
        return None, "output tensor shape unavailable"

    # TFLite Conv2D filter layout is [output_channels, filter_height,
    # filter_width, input_channels_per_group].
    if op_name == "CONV_2D":
        if len(valid_inputs) < 2:
            return None, "input/filter shape unavailable"
        out_shape = valid_outputs[0]
        filter_shape = valid_inputs[1]
        if len(out_shape) != 4 or len(filter_shape) != 4:
            return None, f"unsupported Conv2D shapes: out={out_shape}, filter={filter_shape}"
        macs = product(out_shape) * filter_shape[1] * filter_shape[2] * filter_shape[3]
        return macs, (
            f"output_elements({product(out_shape)}) * "
            f"filter_H({filter_shape[1]}) * filter_W({filter_shape[2]}) * "
            f"filter_input_channels({filter_shape[3]})"
        )

    # TFLite DepthwiseConv2D filter layout is [1, filter_height,
    # filter_width, output_channels]. Each output element uses one filter
    # coefficient per spatial position.
    if op_name == "DEPTHWISE_CONV_2D":
        if len(valid_inputs) < 2:
            return None, "input/filter shape unavailable"
        out_shape = valid_outputs[0]
        filter_shape = valid_inputs[1]
        if len(out_shape) != 4 or len(filter_shape) != 4:
            return None, f"unsupported DepthwiseConv2D shapes: out={out_shape}, filter={filter_shape}"
        macs = product(out_shape) * filter_shape[1] * filter_shape[2]
        return macs, (
            f"output_elements({product(out_shape)}) * "
            f"filter_H({filter_shape[1]}) * filter_W({filter_shape[2]})"
        )

    # TFLite FullyConnected weight layout is [units, input_size].
    # Number of MACs = number of output elements * flattened input width.
    if op_name == "FULLY_CONNECTED":
        if len(valid_inputs) < 2:
            return None, "input/weight shape unavailable"
        out_shape = valid_outputs[0]
        weight_shape = valid_inputs[1]
        if len(weight_shape) != 2 or not out_shape:
            return None, f"unsupported FullyConnected shapes: out={out_shape}, weights={weight_shape}"
        input_size = weight_shape[1]
        macs = product(out_shape) * input_size
        return macs, f"output_elements({product(out_shape)}) * input_size({input_size})"

    # For TFLite BATCH_MATMUL, count multiply-accumulates using the inner
    # dimension K and the number of output elements. This covers ordinary
    # dense batched matrix multiplication; broadcasted batch dimensions are
    # represented by the output shape.
    if op_name == "BATCH_MATMUL":
        if len(valid_inputs) < 2:
            return None, "input tensor shapes unavailable"
        lhs, rhs = valid_inputs[0], valid_inputs[1]
        out_shape = valid_outputs[0]
        if len(lhs) < 2 or len(rhs) < 2 or not out_shape:
            return None, f"unsupported BatchMatMul shapes: lhs={lhs}, rhs={rhs}, out={out_shape}"
        adj_x = False
        adj_y = False
        # Option flags are not available in this helper; for standard ResNet
        # graphs the operator is not expected. Infer K from the usual layout.
        k = lhs[-1]
        if k != rhs[-2]:
            if lhs[-2] == rhs[-2]:
                k = lhs[-2]
            else:
                return None, f"cannot infer BatchMatMul reduction dimension: lhs={lhs}, rhs={rhs}"
        return product(out_shape) * k, f"output_elements({product(out_shape)}) * reduction_dimension({k})"

    return 0, "not included in MAC total"


def parse_model(model_path: Path, schema: Any) -> tuple[list[dict[str, Any]], int, Counter]:
    model_bytes = model_path.read_bytes()
    model = schema.Model.GetRootAsModel(model_bytes, 0)
    if model.SubgraphsLength() == 0:
        raise RuntimeError("The TFLite model contains no subgraphs.")

    records: list[dict[str, Any]] = []
    total_macs = 0
    op_totals: Counter = Counter()

    # Most inference models have one main subgraph. Analyze every subgraph,
    # but label it so the report does not silently merge them.
    for sg_idx in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(sg_idx)
        for op_idx in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(op_idx)
            opcode = model.OperatorCodes(op.OpcodeIndex())
            # Newer schema has DeprecatedBuiltinCode and BuiltinCode. Older
            # schema versions may expose only DeprecatedBuiltinCode.
            try:
                code = int(opcode.BuiltinCode())
            except Exception:
                code = int(opcode.DeprecatedBuiltinCode())
            op_name = builtin_operator_name(schema, code)

            in_indices = operator_inputs(op)
            out_indices = operator_outputs(op)
            in_shapes = [get_tensor_shape(subgraph, i) for i in in_indices if i >= 0]
            out_shapes = [get_tensor_shape(subgraph, i) for i in out_indices if i >= 0]

            macs, formula = calculate_operator_macs(op_name, in_shapes, out_shapes)
            if macs is None:
                mac_text: Any = ""
            else:
                mac_text = int(macs)
                if macs > 0:
                    total_macs += int(macs)
                    op_totals[op_name] += int(macs)

            records.append({
                "subgraph": sg_idx,
                "operator_index": op_idx,
                "operator": op_name,
                "input_shapes": " ; ".join(str(s) for s in in_shapes),
                "output_shapes": " ; ".join(str(s) for s in out_shapes),
                "macs": mac_text,
                "formula": formula,
            })

    return records, total_macs, op_totals


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subgraph", "operator_index", "operator",
        "input_shapes", "output_shapes", "macs", "formula",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_markdown(
    path: Path,
    model_path: Path,
    records: list[dict[str, Any]],
    total_macs: int,
    op_totals: Counter,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count_by_operator = Counter(row["operator"] for row in records)
    unknown_rows = [row for row in records if row["macs"] == ""]
    counted_ops = {name for name, count in op_totals.items() if count > 0}

    lines = [
        "# TFLite Model MAC Analysis",
        "",
        f"- **Model:** `{model_path.as_posix()}`",
        f"- **Total counted MACs:** `{total_macs:,}`",
        "- **Counting convention:** one multiplication plus its accumulation is counted as one MAC.",
        "- **Operators included in the total:** `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, and `BATCH_MATMUL` when tensor shapes are recognized.",
        "- **Operators not included:** elementwise operations (e.g. ADD), pooling, activation, reshape, padding, quantization/requantization, and other non-listed operators.",
        "- **Scope:** static calculation from the TFLite FlatBuffer graph and tensor shapes; it is not a dynamic instruction count.",
        "",
        "## MACs by operator type",
        "",
        "| Operator | Operator count | MACs | Share of counted MACs |",
        "|---|---:|---:|---:|",
    ]
    for name, macs in sorted(op_totals.items(), key=lambda item: item[1], reverse=True):
        count = count_by_operator[name]
        share = (macs / total_macs * 100.0) if total_macs else 0.0
        lines.append(f"| `{name}` | {count} | {macs:,} | {share:.4f}% |")
    lines.extend([
        "",
        "## Operator-level details",
        "",
        "The complete per-operator record, including input/output shapes and formulas, is in `model_mac_breakdown.csv`.",
        "",
        "## Coverage notes",
        "",
        f"- Operators in graph: {len(records)}",
        f"- Operator types in graph: {len(count_by_operator)}",
        f"- Operator types contributing to MAC total: {', '.join(f'`{x}`' for x in sorted(counted_ops)) if counted_ops else 'none'}",
    ])
    if unknown_rows:
        lines.append(
            f"- Rows with unrecognized/insufficient shapes: {len(unknown_rows)}. Their MAC values are blank in the CSV and excluded from the total; review these rows before treating the total as complete."
        )
    else:
        lines.append("- No supported MAC operator had an unrecognized shape.")
    lines.extend([
        "",
        "## Formula conventions",
        "",
        "- `CONV_2D`: `output_elements × filter_height × filter_width × input_channels_per_group`.",
        "- `DEPTHWISE_CONV_2D`: `output_elements × filter_height × filter_width`.",
        "- `FULLY_CONNECTED`: `output_elements × weight_input_size`.",
        "- `BATCH_MATMUL`: `output_elements × reduction_dimension` when the reduction dimension can be inferred from tensor shapes.",
        "",
        "Generated by `scripts/analyze_model_macs.py`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Count MACs from a TFLite model graph.")
    parser.add_argument(
        "--model", type=Path,
        default=repo_root / "src/model/resnet18_int8_channelwise.tflite",
        help="Path to the .tflite model (default: project ResNet-18 model).",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=repo_root / "experiments/B0_measurement",
        help="Directory for CSV and Markdown outputs.",
    )
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not model_path.is_file():
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        return 2

    try:
        schema = load_tflite_schema()
        records, total_macs, op_totals = parse_model(model_path, schema)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    csv_path = out_dir / "model_mac_breakdown.csv"
    md_path = out_dir / "model_mac_summary.md"
    write_csv(csv_path, records)
    write_markdown(md_path, model_path.relative_to(repo_root) if model_path.is_relative_to(repo_root) else model_path,
                   records, total_macs, op_totals)

    print("TFLite MAC analysis complete")
    print(f"Model: {model_path}")
    print(f"Total counted MACs: {total_macs:,}")
    print("MACs by operator type:")
    for name, macs in sorted(op_totals.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name:24s} {macs:>18,}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Operators analyzed: {len(records)}")
    unknown = sum(1 for row in records if row["macs"] == "")
    if unknown:
        print(f"WARNING: {unknown} operator rows have unknown MAC counts; inspect the CSV/Markdown report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



