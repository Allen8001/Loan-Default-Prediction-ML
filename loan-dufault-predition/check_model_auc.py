import joblib

from src.paths import iter_model_paths, setup_project_imports

setup_project_imports()

print("Available models and their stored AUC scores:")
print("=" * 60)
for model_path in iter_model_paths():
    try:
        data = joblib.load(model_path)
        auc = data.get('auc_score', 'N/A')
        print(f"{model_path.name:50s} ({model_path.parent.name}) AUC = {auc}")
    except Exception as e:
        print(f"{model_path.name:50s} Error: {e}")
