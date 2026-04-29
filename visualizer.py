from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ModelInfo:
	learning_rate: float | None = None
	epochs: int | None = None
	input_size: int | None = None
	hidden_size: int | None = None
	output_size: int | None = None
	hidden_activation: str | None = None
	output_activation: str | None = None
	train_samples: int | None = None

	def parameter_count(self) -> int | None:
		if self.input_size is None or self.hidden_size is None or self.output_size is None:
			return None
		# w1: input*hidden, b1: hidden, w2: hidden*output, b2: output
		return (
			self.input_size * self.hidden_size
			+ self.hidden_size
			+ self.hidden_size * self.output_size
			+ self.output_size
		)


@dataclass(frozen=True)
class Metrics:
	n: int
	mse: float
	rmse: float
	mae: float
	r2: float
	max_abs_error: float
	accuracy_at_tol: float


def _mean(values: Iterable[float]) -> float:
	values = list(values)
	if not values:
		raise ValueError("Cannot compute mean of empty sequence")
	return sum(values) / len(values)


def compute_metrics(y_true: list[float], y_pred: list[float], tol: float) -> Metrics:
	if len(y_true) != len(y_pred):
		raise ValueError(f"Length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}")
	if len(y_true) == 0:
		raise ValueError("No rows found in CSV")

	errors = [p - t for t, p in zip(y_true, y_pred)]
	abs_errors = [abs(e) for e in errors]
	sq_errors = [e * e for e in errors]

	mse = _mean(sq_errors)
	rmse = math.sqrt(mse)
	mae = _mean(abs_errors)
	max_abs_error = max(abs_errors)

	y_bar = _mean(y_true)
	ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
	ss_tot = sum((t - y_bar) ** 2 for t in y_true)
	r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

	within = sum(1 for ae in abs_errors if ae <= tol)
	accuracy_at_tol = 100.0 * within / len(abs_errors)

	return Metrics(
		n=len(y_true),
		mse=mse,
		rmse=rmse,
		mae=mae,
		r2=r2,
		max_abs_error=max_abs_error,
		accuracy_at_tol=accuracy_at_tol,
	)


def load_predictions_csv(csv_path: Path) -> tuple[list[float], list[float], list[float]]:
	x_vals: list[float] = []
	y_true: list[float] = []
	y_pred: list[float] = []

	with csv_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		required = {"x", "sin_x", "predicted"}
		missing = required - set(reader.fieldnames or [])
		if missing:
			raise ValueError(f"CSV missing required columns: {sorted(missing)}")

		for row in reader:
			x_vals.append(float(row["x"]))
			y_true.append(float(row["sin_x"]))
			y_pred.append(float(row["predicted"]))

	return x_vals, y_true, y_pred


def parse_cpp_hyperparams(cpp_path: Path) -> ModelInfo:
	if not cpp_path.exists():
		return ModelInfo()

	text = cpp_path.read_text(encoding="utf-8", errors="replace")

	def _re_float(pattern: str) -> float | None:
		m = re.search(pattern, text)
		return None if m is None else float(m.group(1))

	def _re_int(pattern: str) -> int | None:
		m = re.search(pattern, text)
		return None if m is None else int(m.group(1))

	learning_rate = _re_float(r"#define\s+LEARNING_RATE\s+([0-9]*\.?[0-9]+)f?")
	epochs = _re_int(r"#define\s+EPOCHS\s+(\d+)")

	input_size = _re_int(r"\bint\s+input_size\s*=\s*(\d+)\s*;")
	hidden_size = _re_int(r"\bint\s+hidden_size\s*=\s*(\d+)\s*;")
	output_size = _re_int(r"\bint\s+output_size\s*=\s*(\d+)\s*;")

	train_samples = _re_int(r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*(\d+)\s*;\s*\+\+i\s*\)\s*\{")

	hidden_activation = None
	output_activation = None
	if re.search(r"\ba1\s*=\s*tanhActivation\(", text):
		hidden_activation = "tanh"
	if re.search(r"\ba2\s*=\s*sigmoid\(", text):
		output_activation = "sigmoid"

	return ModelInfo(
		learning_rate=learning_rate,
		epochs=epochs,
		input_size=input_size,
		hidden_size=hidden_size,
		output_size=output_size,
		hidden_activation=hidden_activation,
		output_activation=output_activation,
		train_samples=train_samples,
	)


def format_summary(metrics: Metrics, model: ModelInfo, tol: float) -> str:
	parts: list[str] = []
	parts.append(f"N={metrics.n}")
	parts.append(f"MSE={metrics.mse:.6f}")
	parts.append(f"RMSE={metrics.rmse:.6f}")
	parts.append(f"MAE={metrics.mae:.6f}")
	parts.append(f"R^2={metrics.r2:.6f}" if not math.isnan(metrics.r2) else "R^2=nan")
	parts.append(f"Max|err|={metrics.max_abs_error:.6f}")
	parts.append(f"Acc@{tol:g}={metrics.accuracy_at_tol:.1f}%")

	# Second line: model/training details (when available)
	model_bits: list[str] = []
	if model.input_size is not None and model.hidden_size is not None and model.output_size is not None:
		model_bits.append(f"Arch {model.input_size}-{model.hidden_size}-{model.output_size}")
		pcount = model.parameter_count()
		if pcount is not None:
			model_bits.append(f"Params={pcount}")
	if model.hidden_activation or model.output_activation:
		ha = model.hidden_activation or "?"
		oa = model.output_activation or "?"
		model_bits.append(f"Act {ha}/{oa}")
	if model.learning_rate is not None:
		model_bits.append(f"LR={model.learning_rate:g}")
	if model.epochs is not None:
		model_bits.append(f"Epochs={model.epochs}")
	if model.train_samples is not None:
		model_bits.append(f"TrainN={model.train_samples}")

	return " | ".join(parts) + ("\n" + " | ".join(model_bits) if model_bits else "")


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Visualize predictions.csv and print resume-friendly metrics "
			"(MSE/RMSE/MAE/R^2 + tolerance-accuracy), optionally parsing hyperparams from NeuralNetwork.cpp."
		)
	)
	parser.add_argument("--csv", default="predictions.csv", help="Path to predictions CSV")
	parser.add_argument("--cpp", default="NeuralNetwork.cpp", help="Path to C++ source for hyperparam parsing")
	parser.add_argument("--tol", type=float, default=0.05, help="Tolerance for Acc@tol (absolute error)")
	parser.add_argument("--save", default="results.png", help="Output plot path")
	parser.add_argument("--no-show", action="store_true", help="Do not display plot window")

	args = parser.parse_args()
	csv_path = Path(args.csv)
	cpp_path = Path(args.cpp)
	save_path = Path(args.save)

	x_vals, y_true, y_pred = load_predictions_csv(csv_path)
	metrics = compute_metrics(y_true=y_true, y_pred=y_pred, tol=args.tol)
	model = parse_cpp_hyperparams(cpp_path)
	summary = format_summary(metrics=metrics, model=model, tol=args.tol)

	print("\n=== Model Evaluation Summary ===")
	print(summary)
	print(f"CSV: {csv_path}")
	if cpp_path.exists():
		print(f"Parsed params from: {cpp_path}")
	print("===============================\n")

	plt.figure(figsize=(11, 6))
	plt.plot(x_vals, y_true, label="sin(x)", linewidth=2)
	plt.plot(x_vals, y_pred, label="Predicted", linestyle="--", linewidth=2)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Neural Network Approximation of sin(x)")
	plt.legend(loc="upper right")
	plt.grid(True)

	plt.gca().text(
		0.02,
		0.02,
		summary,
		transform=plt.gca().transAxes,
		fontsize=9,
		verticalalignment="bottom",
		bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
	)

	save_path.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(save_path, dpi=200)

	if not args.no_show:
		plt.show()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
