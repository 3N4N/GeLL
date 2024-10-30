hf_access_token = "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"

import os, sys
import random
sys.path.append(os.path.abspath('../evaluate'))
sys.path.append(os.path.abspath('../flan-t5'))
from helper import (
    group_update,
    template_update,
    get_GA,
    get_ED,
    plg,
    imitate_logs,
    prepare_data
)



import numpy as np
import pandas as pd
from datetime import datetime

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM,AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
import torch

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )


# model_name='meta-llama/Llama-3.1-8B'
# model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_access_token)

# model_name='meta-llama/Llama-3.1-8B'
# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              quantization_config=bnb_config,
#                                              device_map='auto',
#                                              use_auth_token=hf_access_token)

model_name='google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto')

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

if torch.cuda.is_available():
    use_cuda = True
    DEVICE = 'cuda'
else:
    use_cuda = False
    DEVICE = 'cpu'
print(f"Using cuda: {use_cuda}")

seed = 61
random.seed(seed)
max_num = 3
use_logs = -1
project = 'HDFS'
train_data, parse_data = imitate_logs(project, max_num)

train_dataset = []
for i in train_data[-use_logs*3:]:
    train_dataset.append({
        "content": i[0],
        "template": i[1]
    })

parse_dataset = []
for i in parse_data[-use_logs:]:
    parse_dataset.append({
        "content": i[0],
        "template": i[1]
    })

dataset = {'train': train_dataset, 'test': parse_dataset}


def make_prompt(dataset, example_indices_full, example_index_to_summarize):
    prompt = ''
    for index in example_indices_full:
        content = dataset['train'][index]['content']
        template = dataset['train'][index]['template']

        # The stop sequence '{summary}\n\n\n' is important for FLAN-T5
        # Other models may have their own preferred stop sequence.
        prompt += f"Parse the raw log to log template: {content}\n{template}\n\n\n"

    content = dataset['test'][example_index_to_summarize]['content']

    prompt += f"Parse the raw log to log template: {content}\n"

    return prompt

start_time = datetime.now()

print(f"Project:{project}, Model: {model_name}")
lg = {}
predictions = []
ground_truths = []
for i in range(len(dataset['test'])):
    # if i!=0: continue

    logid = i
    batch = (i+1)%100
    shots = range(logid*max_num,(logid+1)*max_num)
    one_shot_prompt = make_prompt(dataset, shots, logid)
    print(one_shot_prompt)

    content = dataset['test'][logid]['content']
    template = dataset['test'][logid]['template']

    inputs = tokenizer(one_shot_prompt, return_tensors='pt')
    input_ids = inputs["input_ids"].to(DEVICE)
    model_generation = model.generate(input_ids)[0]
    output = tokenizer.decode(
        model_generation,
        # model_generation[input_ids.shape[1]:],
        skip_special_tokens=True
    )
    print(f"\nACTUAL TEMPLATE:\n{template}")
    print(f'\nMODEL GENERATION - FEW SHOT:\n{output}')

    predictions.append(output)
    ground_truths.append(template)
    lg = group_update(project, lg, content, output, logid)

    if (i+1)%100==0:
        GA = get_GA(lg, ground_truths)
        ED,_ = get_ED(predictions, ground_truths)
        print(f"{(i+1)//100:04d},{GA:.4f},{ED:.4f}")

plg(lg)

finish_time = datetime.now()
duration = finish_time - start_time

days = duration.days
hours = duration.seconds // 3600
minutes = (duration.seconds // 60) % 60
seconds = duration.seconds % 60
print(f"\n\nRunning: {days} days, {hours} hours, {minutes} mins, {seconds} sec.")
