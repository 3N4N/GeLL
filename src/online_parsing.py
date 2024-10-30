import os
import torch
import sys
import argparse
import string
import csv
import re
import numbers
import copy
from datetime import datetime
import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from openprompt.pipeline_base import PromptForGeneration
from openprompt.prompts.generation_verbalizer import GenerationVerbalizer

from helper import (
    wordsplit,
    group_update,
    template_update,
    get_GA,
    get_ED,
    prepare_data
)


# args
max_num             = 3
batch_size          = 10
batch_size_train    = 5
model               = 'flan-t5-small'
num_epochs          = 1
learning_rate       = 5e-4
systems             = "Apache"
systems             = "Android,Apache,BGL,HDFS,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Spark,Thunderbird,Windows,Zookeeper"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False
print(f"Using cuda: {use_cuda}")

seed = 61
random.seed(seed)
lr = learning_rate

model_name = "t5"
rootdir = '..'

for project in systems.split(","):
    start_time = datetime.now()
    pretrainedmodel_path = rootdir + "/LLMs/{}/".format(model)
    train_data, parse_data, logs = prepare_data(rootdir, project, max_num)
    print(
        "Project: {}, Seed : {} , Model : {} , Epoch : {} , Batch_size : {} , Learning Rate : {}".format(
            project,
            str(seed),
            pretrainedmodel_path,
            str(num_epochs),
            str(batch_size),
            str(lr),
        )
    )

    # load plm
    from openprompt.plms import load_plm
    from openprompt.plms.seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper
    from openprompt.plms.lm import LMTokenizerWrapper
    from transformers import T5TokenizerFast

    new_words = ["<*>", "{", "}", "<", "\\"]
    plm, tokenizer, model_config, WrapperClass = load_plm(
        model_name, pretrainedmodel_path
    )
    tokenizer = T5TokenizerFast.from_pretrained(pretrainedmodel_path)
    tokenizer.add_tokens(new_tokens=new_words)

    from openprompt.prompts import ManualTemplate

    template_text = (
        'Parse the raw log to log template: {"placeholder":"text_a"}  {"mask"}'
    )
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

    # define the verbalizer
    from openprompt.data_utils.utils import InputExample
    from openprompt.prompts import GenerationVerbalizer

    label_words = {0: ["{'meta':'EventTemplate'}"]}
    myverbalizer = GenerationVerbalizer(
        tokenizer, classes=None, is_rule=True, label_words=label_words
    )

    # define prompt model for classification
    prompt_model = PromptForGeneration(
        plm=plm, template=mytemplate, tokenizer=tokenizer
    )
    if use_cuda:
        prompt_model = prompt_model.cuda()
    # DataLoader
    from openprompt import PromptDataLoader

    def getdataloader(data_list, batch_size, mytemplate, myverbalizer, mytokenizer,
                      wrapperclass, training):
        dataloader = PromptDataLoader(
            dataset=data_list,
            template=mytemplate,
            verbalizer=myverbalizer,
            tokenizer=mytokenizer,
            tokenizer_wrapper_class=wrapperclass,
            max_seq_length=256,
            decoder_max_length=256,
            batch_size=batch_size,
            shuffle=True if training else False,
            teacher_forcing=True if training else False,
            predict_eos_token=True if training else False,
            truncate_method="tail",
        ).dataloader
        return dataloader

    train_dataloader = getdataloader(train_data, batch_size, mytemplate, myverbalizer, tokenizer, WrapperClass, True)
    parse_dataloader = getdataloader(parse_data, batch_size, mytemplate, myverbalizer, tokenizer, WrapperClass, False)

    from transformers import get_linear_schedule_with_warmup
    from torch.optim import AdamW

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in prompt_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in prompt_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    generation_arguments = {
        "max_length": 128,
    }


    tot_loss = 0
    log_loss = 0
    best_val_acc = 0
    glb_step = 0
    actual_step = 0
    leave_training = False

    acc_traces = []
    tot_train_time = 0
    pbar_update_freq = 10

    predictions = []
    ground_truths = []
    lg = {}

    for epoch in range(num_epochs):
        # train
        prompt_model.train()
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            loss = prompt_model(inputs)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    for i, parse_inputs in enumerate(parse_dataloader):
        if use_cuda:
            parse_inputs = parse_inputs.cuda()

        prompt_model.eval()
        _, output_sentences = prompt_model.generate(
            parse_inputs, **generation_arguments, verbose=False
        )
        predictions.extend(output_sentences)
        ground_truths.extend(parse_inputs["tgt_text"])

        for j, prediction in enumerate(output_sentences):
            idx = i*batch_size + j
            lg = group_update(project, lg, logs[idx][0], prediction, idx)

    # with open("preds1.txt", "w") as f:
    #     f.write("\n".join(predictions))

    log_groups = lg

    # logs_split = [wordsplit(log, project) for log in [i[0] for i in logs]]
    # log_groups = template_update(lg, logs_split, predictions)
    # for key in log_groups.keys():
    #     for id in log_groups[key]:
    #         predictions[id] = key

    GA = get_GA(log_groups, ground_truths)
    print("GA:", GA)
    ED,_ = get_ED(predictions, ground_truths)
    print("ED:", ED)

    # with open("preds2.txt", "w") as f:
    #     f.write("\n".join(predictions))

    finish_time = datetime.now()
    duration = finish_time - start_time

    days = duration.days
    hours = duration.seconds // 3600
    minutes = (duration.seconds // 60) % 60
    seconds = duration.seconds % 60

    print(f"\n\nRunning: {days} days, {hours} hours, {minutes} mins, {seconds} sec.")
