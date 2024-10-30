import re
import copy
import csv
import numpy as np
import pandas as pd
import random
from nltk.metrics.distance import edit_distance
from pathlib import Path

seed = 61
random.seed(seed)
np.random.seed(seed)

def random_replace(string):
    '''
        variable → regular expression → imitated variable
    '''
    letters = re.findall('[a-zA-Z]', string)
    numbers = re.findall('[0-9]', string)

    replaced_letters = random.choices(letters, k=len(letters))
    replaced_numbers = random.choices(numbers, k=len(numbers))

    replaced_string = re.sub('[a-zA-Z]', lambda _: replaced_letters.pop(0), string)
    replaced_string = re.sub('[0-9]', lambda _: replaced_numbers.pop(0), replaced_string)

    return replaced_string


def sim(seq1, seq2):
    long  = copy.copy(seq1) if len(seq1) > len(seq2) else copy.copy(seq2)
    short = copy.copy(seq2) if len(seq1) > len(seq2) else copy.copy(seq1)
    simTokens = 0
    for token in short:
        if token in long:
            simTokens+=1
    retVal = simTokens / len(long)
    return retVal


def find_continuous_subsequences(nums):
    subsequences = []
    current_subsequence = []

    for i in range(len(nums)):
        if i > 0 and nums[i] != nums[i - 1] + 1:
            if len(current_subsequence) > 1:
                subsequences.append(current_subsequence)
            current_subsequence = []

        current_subsequence.append(nums[i])

    if len(current_subsequence) > 1:
        subsequences.append(current_subsequence)

    return subsequences

def convert_to_regex_pattern(string):
    regex_pattern = ''
    for char in string:
        if char.isalpha():
            regex_pattern += r'[a-zA-Z]'
        elif char.isdigit():
            regex_pattern += r'-?\d*'
        else:
            if char != '-':
                regex_pattern += re.escape(char)
    return regex_pattern


def is_number(s):
    for func in (float, lambda x: int(x, 0)):
        try:
            func(s)
            return True
        except ValueError:
            continue
    return False

def wordsplit(log,dataset,regx=None,regx_use=False):

    if dataset == 'Android':
        log = re.sub('\(', '( ', log)
        log = re.sub('\)', ') ', log)
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    # elif dataset == 'Apache':
    #     log = re.sub(',', ', ', log)
    elif dataset == 'BGL':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
        log = re.sub('core\.', 'core. ', log)

    elif dataset == 'Hadoop':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        # log = re.sub(',', ', ', log)
    elif dataset == 'HDFS':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'HealthApp':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'HPC':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
        # log = re.sub('-', '- ', log)
        # log = re.sub('\[', '[ ', log)
        # log = re.sub(']', '] ', log)

    elif dataset == 'Linux':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'Mac':
        log = re.sub('\[', '[ ', log)
        log = re.sub(']', '] ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'OpenSSH':
        log = re.sub('=', '= ', log)
        log = re.sub(':', ': ', log)
        log = re.sub(',', ', ', log)
    # elif dataset == 'OpenStack':
    #     log = re.sub(',', ', ', log)
    elif dataset == 'Spark':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
        log = re.sub('/', '/ ', log)

    elif dataset == 'Proxifier':
        log = re.sub('\(.*?\)', '', log)
        log = re.sub(':', ' ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'Thunderbird':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
        log = re.sub('@', '@ ', log)
    elif dataset == 'Windows':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub('\[', '[ ', log)
        log = re.sub(']', '] ', log)
        log = re.sub(',', ', ', log)
    elif dataset == 'Zookeeper':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)

    if regx_use == True:
        for ree in regx:
            log = re.sub(ree, '<*>', log)

    log = re.split(' +', log)
    return log



def wordcomp(prediction, template):
    _template = list(template)
    for i, _token in enumerate(_template):
        token = re.sub(r'\W+', '', _token)
        _template[i] = token
    for i, _token in enumerate(prediction):
        token = re.sub(r'\W+', '', _token)
        if token != '' and token not in _template:
            # print("---")
            # print(token, _template)
            if not is_number(token) and i < len(template) and "<*>" not in template[i]:
                return False
    return True


def has_consecutive_variable(log,key):
    long  = copy.copy(log) if len(log)>len(key) else copy.copy(key)
    short = copy.copy(key) if len(log)>len(key) else copy.copy(log)
    long, short = list(long), list(short)
    save=[]
    for word in long:
        if word not in short:
           save.append(long.index(word))
    consecutive=find_continuous_subsequences(save)
    for i in range(len(consecutive)):
                start=consecutive[i][0]
                end=consecutive[i][len(consecutive[i])-1]
                comppare=0
                for word in long[start:end+1]:
                    if comppare == 0:
                        regex = convert_to_regex_pattern(word)
                        comppare = 1
                    else:
                        if not re.findall(regex, word):
                            return False
                long[start:end + 1] = ['<*>']
    if len(long)==len(short):
        return True
    else:
        return False


def get_templ(template, logsplit):
    template = list(template)
    for i, token in enumerate(template):
        if token not in logsplit:
            template[i] = "<*>"
    # from itertools import groupby
    # template = [key for key, _group in groupby(template)]
    return tuple(template)

def group_update(project, lg, log, prediction, appendvalue):
    # print(prediction)
    pred = prediction.replace("<*>", " ")
    pred = tuple(wordsplit(pred,project))

    # pred = wordsplit(prediction,project)
    # for i,p in enumerate(pred):
    #     if "<*>" in p:
    #         pred[i] = p.replace("<*>", "")

    logsplit = tuple(wordsplit(log,project))
    candidates = []

    # print(appendvalue, log)
    # print(logsplit)
    # print(pred)

    keys = sorted(lg.keys(), key=lambda x: len(x))
    for templ in keys:
        # if wordcomp(pred, templ) or wordcomp(logsplit, templ):
        if wordcomp(logsplit, templ) or wordcomp(pred, templ):
            _templ = get_templ(templ, logsplit)
            if set(_templ) == {'<*>'}: continue
            if not has_consecutive_variable(logsplit, _templ): continue
            # if len(_templ) != len(logsplit): continue
            candidates.append([templ, _templ])

    if len(candidates) >= 1:
        sim_list=[]
        for cand in candidates:
            sim_list.append(sim(cand[1], logsplit))
        maxsim = sim_list.index(max(sim_list))
        templ  = candidates[maxsim][0]
        _templ = candidates[maxsim][1]
        if _templ != templ:
            lg[_templ] = lg.pop(templ)
        lg[_templ].append(appendvalue)
        # print(candidates)
        # print("_templ:", _templ)
        plg(lg)
    else:
        if logsplit not in lg:
            lg[logsplit] = [appendvalue]
        else:
            lg[logsplit].append(appendvalue)
        # print("group_update: else>else")
        plg(lg)

    return lg

def exclude_digits(string):
    pattern = r'\d'
    digits = re.findall(pattern, string)
    if len(digits)==0:
        return False
    return len(digits)/len(string) >= 0.3


def template_update(template_group,log_sentence):
    new_template_group={}
    for key in template_group.keys():
        group_list = template_group[key]
        if group_list[0] == -1:
            continue
        len_save = list()
        for id in group_list:
            lenth = len(log_sentence[id])
            len_save.append(lenth)
        max_lenth = max(len_save, key=len_save.count)
        filter = {}
        for id in group_list:
            group_member = log_sentence[id]
            if len(group_member) != max_lenth:
                continue
            else:
                ind = 0
                for word in group_member:
                    filter.setdefault(ind, []).append(word)
                    ind += 1
        templat = list()
        for key in filter.keys():
            all_words = filter[key]
            all_words = list(set(all_words))
            if len(all_words) == 1:
                if not exclude_digits(all_words[0]):
                    templat.append(all_words[0])
                else:
                    templat.append('<*>')
            else:
                templat.append('<*>')
        templat = ' '.join(templat)
        for id in group_list:
            new_template_group.setdefault(templat, []).append(id)
    return new_template_group


def get_GA(group,ground_list):
    sum = len(ground_list)
    correct = 0
    count=0

    for key in group.keys():
        tag = 0
        predict=group[key]
        predict_group_num=len(predict)
        count+=predict_group_num
        groundtruth_num=ground_list.count(ground_list[predict[0]])
        if predict_group_num==groundtruth_num:
            for i in range(len(predict)):
                if ground_list[predict[i]] != ground_list[predict[0]]:
                    tag=1
            if tag==1:
                continue
            else:
                correct+=predict_group_num
    GA=correct/sum
    return GA

def get_ED(predictions,groundtruths):

    df = pd.DataFrame({'predictions': predictions, 'groundtruths': groundtruths})
    df['predictions_']  = df["predictions"].str.replace( "\s+", "", regex=True)
    df['groundtruths_'] = df["groundtruths"].str.replace( "\s+", "", regex=True)
    edit_distance_result = []
    for i, j in zip(
        np.array(df['predictions_'].values, dtype="str"),
        np.array(df['groundtruths_'].values, dtype="str"),
    ):
        edit_distance_result.append(edit_distance(i, j))

    edit_distance_result_mean = np.mean(edit_distance_result)
    edit_distance_result_std  = np.std(edit_distance_result)

    return edit_distance_result_mean, edit_distance_result_std



def plg(lg):
    # print("{\n" + "\n".join("{!r}: {!r},".format(k, v) for k, v in lg.items()) + "\n}")
    pass


def imitate_logs(rootdir, project, max_num):
    dataset_folder = rootdir + "/logs/" + project
    dataset_path = dataset_folder + "/" + project + "_2k.log_structured_corrected.csv"
    raw_dataset = pd.read_csv(dataset_path)
    raw_dataset = raw_dataset[['Content', 'EventTemplate']]
    raw_dataset = raw_dataset.map( str)  # must convert to string or else will hit error
    raw_dataset = raw_dataset[:100]

    def read_csv_to_list(file_path):
        data_list = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data_list.append(row[0])
        return data_list

    variablelist = read_csv_to_list(rootdir + "/Variableset/variablelist1" + project + ".csv")

    train_data = []
    parse_data = []
    for i in range(len(raw_dataset)):
        last_id = []
        x = raw_dataset.iloc[i,0]
        y = raw_dataset.iloc[i,1]
        parse_data.append([x,y])
        for j in range(max_num):
            xx = wordsplit(x,project) # .split()
            yy = wordsplit(y,project) # .split()
            # print(xx,yy)
            ids = []
            for id in range(len(yy)):
                # if "<*>" not in yy[id]:
                if xx[id] not in variablelist:
                    ids.append(id)
            if len(ids) == 0:
                continue
            idx = random.choice(ids)
            last_id.append(idx)
            replace_variable = variablelist[random.randrange(0, len(variablelist))]
            xx[idx] = replace_variable
            yy[idx] = "<*>"
            _x = " ".join(xx)
            _y = " ".join(yy)
            train_data.append([_x,_y])
            # if j == 0:
            #     parse_data.append([_x,_y])
            # else:
            #     train_data.append([_x,_y])
    return train_data, parse_data


def prepare_data(rootdir, project,max_num):

    train_data, parse_data = imitate_logs(rootdir, project, max_num)

    from openprompt.data_utils import InputExample

    train_dataset = []
    for i in range(len(train_data)):
        input_example = InputExample(
            text_a=train_data[i][0],
            meta={"EventTemplate": train_data[i][1]},
            label=0,
        )
        train_dataset.append(input_example)

    parse_dataset = []
    for i in range(len(parse_data)):
        input_example = InputExample(
            text_a=parse_data[i][0],
            meta={"EventTemplate": parse_data[i][1]},
            label=0,
        )
        parse_dataset.append(input_example)

    templates = [item[1] for item in parse_data]
    ground_list = {}
    ground_list_values = pd.Series(templates).value_counts().index.tolist()
    for value in ground_list_values:
        indices = [i for i, x in enumerate(templates) if x == value]
        ground_list[value] = indices

    print("LEN:", len(train_dataset), len(parse_dataset))
    return train_dataset, parse_dataset, parse_data
