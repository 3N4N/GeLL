import csv
import json
import random
import pandas as pd

from settings import benchmark_settings
from preprocessing import LogLoader, wordsplit

def mutate_logs(rootdir, project, max_num):
    dataset_folder = rootdir / "logs" / project
    dataset_logfile = dataset_folder / (project+"_2k.log")
    dataset_path = dataset_folder / (project+"_2k.log_structured_corrected.csv")

    logloader = LogLoader(benchmark_settings[project]['log_format'])
    log_df = logloader.format(dataset_logfile)

    raw_dataset = pd.read_csv(dataset_path)
    raw_dataset = raw_dataset[['Content', 'EventTemplate']]
    # raw_dataset['Content'] = log_df['Content']
    raw_dataset = raw_dataset.map( str)  # must convert to string or else will hit error
    # raw_dataset = raw_dataset[:10]

    def read_csv_to_list(file_path):
        data_list = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data_list.append(row[0])
        return data_list

    variablelist = read_csv_to_list(rootdir / "Variableset" / ("variablelist1" + project + ".csv"))

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

if __name__ == '__main__':
    import argparse
    import os
    from pathlib import Path
    rootdir = Path(__file__).absolute().parent.parent
    print(f"Project root: {rootdir}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--n_mutations", type=int, default=3)
    args = parser.parse_args()
    print(rootdir, args.project,args.n_mutations)

    for project in benchmark_settings.keys():
        if args.project is not None and project != args.project: continue

        train_data, parse_data = mutate_logs(rootdir, project,args.n_mutations)
        train_df = pd.DataFrame(train_data, columns=['Content', 'Template'])
        parse_df = pd.DataFrame(parse_data, columns=['Content', 'Template'])
        savedir = rootdir / "mutes" / project
        if not os.path.exists(savedir): os.makedirs(savedir)
        train_df.to_csv(savedir / "train.csv", index=None)
        parse_df.to_csv(savedir / "test.csv",  index=None)
