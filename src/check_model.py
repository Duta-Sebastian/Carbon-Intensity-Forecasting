import os
import pickle


def inspect_saved_model(
    model_path="pipeline_output/final_arimax_models/arimax_scaled_dropped_features_specific.pkl",
    output_path="pipeline_output/final_arimax_models/arimax_scaled_dropped_features_specific.txt",
):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"--- Loading model from {model_path} ---")
    with open(model_path, "rb") as f:
        model_instance = pickle.load(f)

    res = model_instance.model_res
    report = []

    report.append(f"--- Model loaded from: {model_path} ---\n")

    report.append("--- Model Summary ---\n")
    report.append(res.summary().as_text())
    report.append("\n")

    report.append("--- Convergence Check ---\n")
    report.append(f"Converged: {res.mle_retvals.get('converged', 'Unknown')}\n")
    report.append(f"Iterations: {res.mle_retvals.get('iterations', 'Unknown')}\n")

    final_text = "\n".join(report)

    print(final_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"\nSaved inspection report to: {output_path}")


if __name__ == "__main__":
    inspect_saved_model()
