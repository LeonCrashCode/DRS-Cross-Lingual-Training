import sys
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import Dataset
import argparse
import numpy as np
import random

def setup_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

device = "cuda"
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--max_input_length', type=int, default=48)
    parser.add_argument('--max_output_length', type=int, default=196)
    parser.add_argument('--num_proc', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--peft_model_id', required=True)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--num_beam_groups', type=int, default=1)
    parser.add_argument('--diversity_penalty', type=float,default=0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--do_sample', action='store_true')
    return parser

setup_seed(seed=123456)

parser = get_parser()
args = parser.parse_args()

print(args)

config = PeftConfig.from_pretrained(args.peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, args.peft_model_id)
model.to(device)
model.eval()

# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

def get_lines(path, lower=False):
    rets = []
    for line in open(path):
        line = line.strip()
        if lower:
            line = line.lower()
        rets.append(line)
    return rets

test_src = get_lines(args.input_file, lower=True)

dataset_test = Dataset.from_dict({"src":test_src})

def preprocess_function(examples):
    inputs = examples["src"]
    model_inputs = tokenizer(inputs, max_length=args.max_input_length, padding="max_length", truncation=True, return_tensors="pt")
    return model_inputs

processed_dataset_test = dataset_test.map(
    preprocess_function,
    batched=True,
    num_proc=args.num_proc,
    remove_columns=dataset_test.column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset dev"
    )
test_dataloader = DataLoader(processed_dataset_test, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)

with open(args.output_file, "w") as f:
    for _, batch in enumerate(tqdm(test_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            outputs = model.generate(**batch, 
                                     max_new_tokens=args.max_output_length, 
                                     num_beams=args.num_beams, 
                                     num_return_sequences=args.num_return_sequences, 
                                     num_beam_groups=args.num_beam_groups,
                                     diversity_penalty=args.diversity_penalty)
        tgt = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
        f.write("\n".join(tgt)+"\n")
        f.flush()
    f.close()




