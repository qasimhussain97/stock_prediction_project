import json
import os
from datetime import datetime

def log_metrics(metrics_table, stock, model_name, rmse, mae):
    metrics_table.append({
        'Stock': stock,
        'Model': model_name, 
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4)
    })

def save_model_metadata(model_dir, template, stock, model_name, rmse, mae): 
    metadata_path = os.path.join(model_dir, template.format(symbol=stock.lower()))
    metadata = {
        "stock" : stock,
        "model_name": model_name, 
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(metadata_path, "w") as f: 
        json.dump(metadata, f, indent = 4)


