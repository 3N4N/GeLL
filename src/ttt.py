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
    group_update,
    template_update,
    get_GA,
    get_ED,
    plg,
    prepare_data
)


# args
max_num             = 3
batch_size          = 100
batch_size_train    = 5
model               = 'flan-t5-small'
num_epochs          = 2
learning_rate       = 5e-4
train_percentage    = "0.025"
projects            = "Spark"
projects            = "Android,Apache,BGL,HDFS,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Spark,Thunderbird,Windows,Zookeeper"

os.remove("result.txt") if os.path.exists("result.txt") else None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False
print(f"Using cuda: {use_cuda}")

seed = 61
random.seed(seed)
np.random.seed(seed)
lr = learning_rate
num_epochs = num_epochs

model_name = "t5"

rootdir = ".."


for project in projects.split(","):
    start_time = datetime.now()

    pretrainedmodel_path = rootdir + "/LLMs/{}/".format(model)
    train_data, parse_data, logs = prepare_data(rootdir, project, max_num)

    # train_data = train_data[-100*max_num:]
    # parse_data = parse_data[-100:]
    # logs = logs[-100:]

    print(
        "Project: {}, Seed : {} , Model : {} , Epoch : {} , Batch_size : {} , Learning Rate : {}".format(
            str(project),
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

    plm2, tokenizer2, model_config2, WrapperClass2 = load_plm(
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

    # define prompt model for classification  # enan: isn't it "for generation"?
    prompt_model = PromptForGeneration(
        plm=plm, template=mytemplate, tokenizer=tokenizer
    )
    prompt_model1 = PromptForGeneration(
        plm=plm, template=mytemplate, tokenizer=tokenizer
    )
    prompt_model2 = PromptForGeneration(
        plm=plm2, template=mytemplate, tokenizer=tokenizer
    )
    if use_cuda:
        prompt_model = prompt_model.cuda()
        prompt_model1 = prompt_model1.cuda()
        prompt_model2 = prompt_model2.cuda()

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

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

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

    generation_arguments = {
        "max_length": 128,
    }



    predictions = []
    ground_truths = []
    predictions2 = []
    ground_truths2 = []
    batch = 0

    best_val_acc = 0

    lg = {}
    lg2 = {}

    for i, parse_inputs in enumerate(parse_dataloader):
        batch += 1

        train_loader = getdataloader(train_data[i*batch_size*max_num:(i+1)*batch_size*max_num],
                                    batch_size_train, mytemplate, None, tokenizer, WrapperClass, True)
        # parse_loader = getdataloader(parse_data[0:(i+1)*batch_size],
        #                             batch_size, mytemplate, None, tokenizer, WrapperClass, False)

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        num_training_steps = num_epochs * ((batch_size*max_num)/batch_size_train)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        best_val_acc = 0
        for epoch in range(num_epochs):
            prompt_model.train()
            for train_inputs in train_loader:
                if use_cuda:
                    train_inputs = train_inputs.cuda()

                loss = prompt_model(train_inputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

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

        if i == 0:
            prompt_model2.load_state_dict(prompt_model.state_dict())

        # prompt_model2.eval()
        # _, output_sentences = prompt_model2.generate(
        #     parse_inputs, **generation_arguments, verbose=False
        # )
        # predictions2.extend(output_sentences)
        # ground_truths2.extend(parse_inputs["tgt_text"])

        # for j, prediction in enumerate(output_sentences):
        #     idx = i*batch_size + j
        #     lg2 = group_update(project, lg2, logs[idx][0], prediction, idx)

        GA = get_GA(lg, ground_truths)
        # GA2 = get_GA(lg2, ground_truths)
        ED,_ = get_ED(predictions, ground_truths)
        # ED2,_ = get_ED(predictions2, ground_truths)
        print(f"{batch:04d},{GA:.4f},{ED:.4f}")
        with open("result.txt", "a") as f:
            print(f"{batch:04d},{GA:.4f},{ED:.4f}", file=f)

    plg(lg)
    # print(lg.keys())

    finish_time = datetime.now()
    duration = finish_time - start_time

    days = duration.days
    hours = duration.seconds // 3600
    minutes = (duration.seconds // 60) % 60
    seconds = duration.seconds % 60

    print(
        f"\n\nRunning: {days} days, {hours} hours, {minutes} mins, {seconds} sec."
    )
