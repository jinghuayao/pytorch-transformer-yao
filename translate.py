from pathlib import Path
from config import get_config, latest_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset
import torch
import sys



def translate(sentence: str):

    # define device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    config = get_config()
    tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer = Tokenizer.from_file()






if __name__ == "__main__":

    # Returns the exact file name (e.g., "script.py")
    current_filename = Path(__file__).name

    # Returns the file name WITHOUT the extension (e.g., "script")
    current_stem = Path(__file__).stem

    print(f"File name: {current_filename}")
    print(f"Just the name: {current_stem}")
    print(f"{current_filename}: run through without any issue.")

    translate(sys.argv[1] if len(sys.argv) > 1 else "I am not a very good student. Hahahahahaha!")




