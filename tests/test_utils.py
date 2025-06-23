import json
import os
from src.utils import log_metrics, save_model_metadata

def test_log_metrics():
    metrics_table = []
    stock = "TEST"
    model_name = "TestModel"
    rmse = 1.23456
    mae = 0.98765

    log_metrics(metrics_table, stock, model_name, rmse, mae)

    assert len(metrics_table) == 1, "Test Failed: The metrics table should have one entry."
    
    entry = metrics_table[0]
    assert isinstance(entry, dict), "Test Failed: The entry should be a dictionary."
    
    expected_keys = ['Stock', 'Model', 'RMSE', 'MAE']
    assert all(key in entry for key in expected_keys), "Test Failed: Dictionary is missing one or more keys."

    assert entry['Stock'] == stock
    assert entry['Model'] == model_name
    assert entry['RMSE'] == 1.2346 
    assert entry['MAE'] == 0.9877  

def test_save_model_metadata(tmp_path):
    model_dir = tmp_path
    template = "metadata_{symbol}.json"
    stock = "TEST"
    model_name = "TestModel"
    rmse = 1.23456
    mae = 0.98765

    save_model_metadata(model_dir, template, stock, model_name, rmse, mae)

    expected_file_path = os.path.join(model_dir, "metadata_test.json")
    
    assert os.path.exists(expected_file_path), "Test Failed: The metadata file was not created."

    with open(expected_file_path, "r") as f:
        metadata = json.load(f)

    assert isinstance(metadata, dict), "Test Failed: The file content is not a valid JSON dictionary."
    

    assert metadata["stock"] == stock
    assert metadata["model_name"] == model_name
    assert metadata["rmse"] == 1.2346
    assert metadata["mae"] == 0.9877
    assert "date" in metadata 