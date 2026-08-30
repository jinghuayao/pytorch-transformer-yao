from pathlib import Path


def get_config():

    config = {
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
         "datasource": "Helsinki-NLP/opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "cache_dir": "./hf_dataset_cache",
        "dataset_fraction": "10%",
    }

    return config

def get_weights_file_path(config, epoch: str):
    # Even if you add deeper subfolders later:
    model_folder = f"model_saved_{config['d_model']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    
    file_path = Path('.') / model_folder / model_filename
    
    # This single line handles the entire upper-level chain
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    return str(file_path)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"model_saved_{config['d_model']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

if __name__ == "__main__":

    # Returns the exact file name (e.g., "script.py")
    current_filename = Path(__file__).name

    # Returns the file name WITHOUT the extension (e.g., "script")
    current_stem = Path(__file__).stem

    print(f"File name: {current_filename}")
    print(f"Just the name: {current_stem}")
    print(f"{current_filename}: run through without any issue.")
