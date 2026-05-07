"""
Data Loader - Load and process the Anywhere dataset
"""

import pandas as pd
import os

def load_dataset():
    """Load the Anywhere dataset from CSV"""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "MaiStorage.csv")
    
    df = pd.read_csv(csv_path)
    return df


def get_summary_stats(df):
    from models.predictor import PredictiveModel
    
    try:
        model = PredictiveModel()
        # Train all models (default) and let the trainer select the best one
        results = model.train_all(df.copy(), verbose=False)
        best_name = model.best_model_name
        if best_name and best_name in results:
            model_acc = results[best_name]["accuracy"] * 100
        else:
            model_acc = 0.0
    except Exception:
        model_acc = 0.0

    stats = {
        'total':        len(df),
        'failures':     int(df['Machine failure'].sum()) if 'Machine failure' in df.columns else 0,
        'failure_rate': (df['Machine failure'].sum() / len(df) * 100) if 'Machine failure' in df.columns else 0,
        'model_acc':    round(model_acc, 1),
    }
    return stats
