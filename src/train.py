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
import sys
import argparse
from peft import PeftModel, PeftConfig

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prefix', required=True)
    parser.add_argument('--dev_prefix', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--save_model_name', required=True)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_input_length', type=int, default=48)
    parser.add_argument('--max_output_length', type=int, default=196)
    parser.add_argument('--num_proc', type=int, default=10)
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--peft_model_id', type=str, default=None)
    return parser

parser = get_parser()
args = parser.parse_args()

print(args)

if args.peft_model_id:
    peft_config = PeftConfig.from_pretrained(args.peft_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, args.peft_model_id, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
else:
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=args.rank, lora_alpha=32, lora_dropout=0.1)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)


model.print_trainable_parameters()

def get_lines(path, lower=False):
    rets = []
    for line in open(path):
        line = line.strip()
        if lower:
            line = line.lower()
        rets.append(line)
    return rets

train_src = get_lines(args.train_prefix+".src", lower=True)
train_tgt = get_lines(args.train_prefix+".tgt")

dev_src = get_lines(args.dev_prefix+".src", lower=True)
dev_tgt = get_lines(args.dev_prefix+".tgt")

dataset_train = Dataset.from_dict({"src":train_src, "tgt":train_tgt})
dataset_dev = Dataset.from_dict({"src":dev_src, "tgt":dev_tgt})

print(dataset_train)
print(dataset_dev)

device = "cuda"

# data preprocessing
def preprocess_function(examples):
    inputs = examples["src"]
    targets = examples["tgt"]
    model_inputs = tokenizer(inputs, max_length=args.max_input_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=args.max_output_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

processed_dataset_train = dataset_train.map(
    preprocess_function,
    batched=True,
    num_proc=args.num_proc,
    remove_columns=dataset_train.column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset train"
    )

processed_dataset_dev = dataset_dev.map(
    preprocess_function,
    batched=True,
    num_proc=args.num_proc,
    remove_columns=dataset_dev.column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset dev"
    )

print(processed_dataset_train[0])

train_dataloader = DataLoader(
        processed_dataset_train, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True
        )

eval_dataloader = DataLoader(processed_dataset_dev, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epoches)
        )

model = model.to(device)

for epoch in range(args.num_epoches+1):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    #eval_preds = []

    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        #eval_preds.extend(
        #        tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        #        )
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

    if epoch % 5 == 0:
        peft_model_id = f"models_{args.save_model_name}/{args.rank}_{args.lr}/{args.save_model_name}_{peft_config.peft_type}_{peft_config.task_type}_{epoch}"
        model.save_pretrained(peft_model_id)
