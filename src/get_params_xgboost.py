import pickle
from pathlib import Path

folder_path = "pipeline_output_xgboost/models"

for file_path in Path(folder_path).glob("*.pkl"):
    print(f"\n--- Hyperparameters for: {file_path.name} ---")

    try:
        with open(file_path, "rb") as file:
            custom_wrapper = pickle.load(file)

        if hasattr(custom_wrapper, "model_params"):
            hyperparameters = custom_wrapper.model_params
            for param, value in hyperparameters.items():
                print(f"{param}: {value}")
        booster = custom_wrapper.model.get_booster()

        importance_weight = booster.get_score(importance_type="weight")
        importance_gain = booster.get_score(importance_type="gain")

        print("--- Feature Weights (Frequency of splits) ---")
        for feature, score in importance_weight.items():
            print(f"{feature}: {score}")

        print("\n--- Feature Gains (Impact on accuracy) ---")
        for feature, score in importance_gain.items():
            print(f"{feature}: {score:.4f}")

    except AttributeError:
        print(
            "Error: This model might be a native XGBoost Booster, which doesn't support .get_params()."
        )
    except Exception as e:
        print(f"Error loading file: {e}")
