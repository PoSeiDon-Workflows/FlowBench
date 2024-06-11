# %% [markdown]
# # LoRA - Parameter-Efficient Fine Tuning
#
# * LoRA on MistralAI-7b-v1 model

# %%
import argparse
import logging
import pickle
from ast import arg
from collections import defaultdict
from datetime import datetime

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)

from flowbench.metrics import (eval_accuracy, eval_average_precision, eval_f1,
                               eval_recall, eval_roc_auc)
# from data_processing import build_text_data, load_tabular_data
from flowbench.utils import create_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

# import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="1000genome")
args = parser.parse_args()

wf = args.dataset
data_folder = "./data"
raw_dataset = load_dataset("csv",
                           data_files={"train": f"{data_folder}/{wf}/train.csv",
                                       "validation": f"{data_folder}/{wf}/validation.csv",
                                       "test": f"{data_folder}/{wf}/test.csv"})

# %%
ckp = "bert-base-uncased"
# ckp = "xlnet-base-cased"
# ckp = "xlnet-large-cased"
# ckp = "meta-llama/Llama-2-7b-hf"
# ckp = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(ckp)
# for llama/mistralai models
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForSequenceClassification.from_pretrained(ckp)


def tokenize_function(data):
    return tokenizer(data["text"], truncation=True)


tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
model

# %%


# def compute_metrics(eval_preds):
#     metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

def compute_metrics(eval_preds):
    logits, y_true = eval_preds
    y_pred = np.argmax(logits, axis=1)
    acc = eval_accuracy(y_true, y_pred)
    f1 = eval_f1(y_true, y_pred)
    auc = eval_roc_auc(y_true, y_pred)
    ap = eval_average_precision(y_true, y_pred)
    recall = eval_recall(y_true, y_pred)
    return {"accuracy": acc,
            "f1": f1,
            "roc_auc": auc,
            "average_precision": ap,
            "recall": recall}

# %% [markdown]
# ## SFT without LoRA


# %%
# setup trainingarguments
# training_args = TrainingArguments(
#     output_dir=f"../models/{wf}/{ckp}",
#     learning_rate=1e-2,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     num_train_epochs=3,
#     weight_decay=1e-4,
#     evaluation_strategy="no",
#     save_strategy="no",
#     load_best_model_at_end=True,
# )

# # setup trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# # train model
# trainer.train()

# # evaluate with test split
# trainer.evaluate(tokenized_datasets["test"])

# ## LoRA
#
# * instantiate a base model
# * create a configuration (`LoRAConfig`) where you define LoRA-specific parameters
# * Wrap the base model with `get_peft_model()` to get trainable `PeftModel`
# * Train the `PeftModel` with `Trainer`

# %% [markdown]
# ## LoRAConfig
#
# * `r`: the rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters.
# * `target_modules`: The modules (for example, attention blocks) to apply the LoRA update matrices.
# * `lora_alpha`: LoRA scaling factor.
# * `bias`: Specifies if the bias parameters should be trained. Can be 'none', 'all' or 'lora_only'.
# * `use_rslora`: When set to True, uses Rank-Stabilized LoRA which sets the adapter scaling factor to lora_alpha/math.sqrt(r), since it was proven to work better. Otherwise, it will use the original default value of lora_alpha/r.
# * `modules_to_save`: List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. These typically include model’s custom head that is randomly initialized for the fine-tuning task.
# * `layers_to_transform`: List of layers to be transformed by LoRA. If not specified, all layers in target_modules are transformed.
# * `layers_pattern`: Pattern to match layer names in target_modules, if layers_to_transform is specified. By default PeftModel will look at common layer pattern (layers, h, blocks, etc.), use it for exotic and custom models.
# * `rank_pattern`: The mapping from layer names or regexp expression to ranks which are different from the default rank specified by r.
# * `alpha_pattern`: The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by lora_alpha.

# %%
if ckp.startswith("xlnet"):
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
                             target_modules=['layer_1', 'layer_2', 'summary'])
elif ckp.startswith("meta-llama"):
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
                             target_modules=["q_proj",
                                             "k_proj",
                                             "v_proj",
                                             "o_proj",
                                             "gate_proj",
                                             "up_proj",
                                             "down_proj"])
elif ckp.startswith("mistralai"):
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
                             target_modules=["q_proj",
                                             "k_proj",
                                             "v_proj",
                                             "o_proj",
                                             "gate_proj",
                                             "up_proj",
                                             "down_proj"])
else:
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# %% [markdown]
# ## LoRA Training

# %%
training_args = TrainingArguments(
    output_dir=f"./models/{wf}/{ckp}-lora",
    # learning_rate=1e-2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=1e-4,
    evaluation_strategy="no",
    save_strategy="no",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

train_result = trainer.train()
train_runtime = train_result.metrics['train_runtime']
test_result = trainer.evaluate(tokenized_datasets["test"])

# %%
# print(f'Training runtime: {train_runtime} seconds')
# print(train_result)
# %%

# print(test_result)
# %%
print(f"{wf} - LLM:",
      f"Training time {train_runtime:.3f}",
      f"Accuracy {test_result['eval_accuracy']:.3f}",
      f"AP {test_result['eval_average_precision']:.3f}",
      f"AUC {test_result['eval_roc_auc']:.3f}",
      f"Recall {test_result['eval_recall']:.3f}")

# %%
