from pathlib import Path
rootdir = Path(__file__).absolute().parent.parent
print(f"Project root: {rootdir}")

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

from openprompt.pipeline_base import PromptForGeneration
from openprompt.prompts.generation_verbalizer import GenerationVerbalizer

from common import (
    wordsplit,
    group_update,
    template_update,
    plg,
    prepare_data
)
from evaluator import (
    get_GA,
    get_ED,
)

parser = argparse.ArgumentParser()
parser.add_argument("--max_num", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--model", type=str, default="flan-t5-small")
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument(
    "--projects",
    type=str,
    default="Android,Apache,BGL,HDFS,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Spark,Thunderbird,Windows,Zookeeper"
    # default="Windows,Apache,Spark",
)
args = parser.parse_args()

# args
max_num             = args.max_num
batch_size          = args.batch_size
model               = args.model
num_epochs          = args.num_epochs
learning_rate       = args.learning_rate
projects            = args.projects
batch_size_training = 5

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

for project in projects.split(","):
    start_time = datetime.now()

    pretrainedmodel_path = rootdir / "LLMs" / model
    train_data, parse_data, logs = prepare_data(rootdir, project)

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
            shuffle=training,
            teacher_forcing=training,
            predict_eos_token=training,
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
                                    batch_size_training, mytemplate, None, tokenizer, WrapperClass, True)
        # parse_loader = getdataloader(parse_data[0:(i+1)*batch_size],
        #                             batch_size, mytemplate, None, tokenizer, WrapperClass, False)

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        num_training_steps = num_epochs * ((batch_size*max_num)/batch_size_training)
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

        # logs_split = [wordsplit(log, project) for log in [i[0] for i in logs]]
        # log_groups = template_update(lg, logs_split)
        log_groups = lg

        for key in log_groups.keys():
            for idx in log_groups[key]:
                predictions[idx] = " ".join(key).replace(sub_sign.strip(),"<*>").strip()
        output_dir = rootdir / "output" / model / project
        output_dir = rootdir / "output" / (model + "-old") / project
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(zip([x[0] for x in logs], predictions, ground_truths))
        df.to_csv(output_dir / "prediction.csv", index=False, header=False)

    plg(log_groups)

    finish_time = datetime.now()
    duration = finish_time - start_time

    days = duration.days
    hours = duration.seconds // 3600
    minutes = (duration.seconds // 60) % 60
    seconds = duration.seconds % 60

    print(
        f"\n\nRunning: {days} days, {hours} hours, {minutes} mins, {seconds} sec."
    )
