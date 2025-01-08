import numpy as np
import pandas as pd
import scipy.special
from nltk.metrics.distance import edit_distance as calc_edit_distance

from settings import benchmark_settings

pd.options.display.float_format = "{:,.2f}".format

def get_accuracy(series_groundtruth, series_parsedlog, filter_templates=None):
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])
    grouped_df = df_combined.groupby('groundtruth')
    accurate_events = 0 # determine how many lines are correctly parsed
    accurate_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
    # for ground_truthId in series_groundtruth_valuecounts.index:
    for ground_truthId, group in grouped_df:
        # logIds = series_groundtruth[series_groundtruth == ground_truthId].index
        series_parsedlog_logId_valuecounts = group['parsedlog'].value_counts()
        if filter_templates is not None and ground_truthId in filter_templates:
            for parsed_eventId in series_parsedlog_logId_valuecounts.index:
                filter_identify_templates.add(parsed_eventId)
        if series_parsedlog_logId_valuecounts.size == 1:
            parsed_eventId = series_parsedlog_logId_valuecounts.index[0]
            if len(group) == series_parsedlog[series_parsedlog == parsed_eventId].size:
                if (filter_templates is None) or (ground_truthId in filter_templates):
                    accurate_events += len(group)
                    accurate_templates += 1
    # print("filter templates: ", len(filter_templates))
    # print("total messages: ", len(series_groundtruth[series_groundtruth.isin(filter_templates)]))
    # print("set of total: ", len(filter_identify_templates))
    # print(accurate_events, accurate_templates)
    if filter_templates is not None:
        GA = float(accurate_events) / len(series_groundtruth[series_groundtruth.isin(filter_templates)])
        PGA = float(accurate_templates) / len(filter_identify_templates)
        RGA = float(accurate_templates) / len(filter_templates)
    else:
        GA = float(accurate_events) / len(series_groundtruth)
        PGA = float(accurate_templates) / len(series_parsedlog_valuecounts)
        RGA = float(accurate_templates) / len(series_groundtruth_valuecounts)
    # print(FGA, RGA)
    FGA = 0.0
    if PGA != 0 or RGA != 0:
        FGA = 2 * (PGA * RGA) / (PGA + RGA)
    return GA, FGA, PGA, RGA

def get_PA(series_groundtruth, series_parsedlog):
    correctly_parsed_logs = series_groundtruth.eq(series_parsedlog).values.sum()
    total_logs = len(series_groundtruth)

    PA = float(correctly_parsed_logs) / total_logs
    return PA

def get_ED(series_groundtruth, series_parsedlog):
    y_true = np.array(series_groundtruth.values, dtype="str")
    y_pred = np.array(series_parsedlog.values, dtype="str")

    edit_distances = [calc_edit_distance(i,j) for (i,j) in zip(y_true, y_pred)]
    return np.mean(edit_distances), np.std(edit_distances)

if __name__ == "__main__":

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--projects",
        type=str,
        default="Android,Apache,BGL,HDFS,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Spark,Thunderbird,Windows,Zookeeper"
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    for project in args.projects.split(","):
        print(project)

        predic_file = Path(args.dirpath) / project / "prediction.csv"
        result_file = Path(args.dirpath) / project / "result.csv"
        df_parsedlog = pd.read_csv(
            predic_file, index_col=False, header=None, names=["Log", "Predict", "EventTemplate"]
        ).map(str)
        f_result = open(result_file, "w")
        print("Batch,GA,PA,ED", file=f_result)

        results = []
        edit_distance_in_batch = []
        for i in range(len(df_parsedlog)//batch_size):
            df = df_parsedlog.iloc[:(i+1)*batch_size].copy()
            series_parsedlog = df["Predict"].str.replace( r"\s+", "", regex=True)
            series_groundtruth = df["EventTemplate"].str.replace( r"\s+", "", regex=True)

            GA, FGA, _,_ = get_accuracy(series_groundtruth, series_parsedlog)
            PA = get_PA(series_groundtruth, series_parsedlog)

            edit_distance,_ = get_ED(series_groundtruth.tail(batch_size),
                                     series_parsedlog.tail(batch_size))
            edit_distance_in_batch.append(edit_distance)
            ED = np.mean(edit_distance_in_batch)

            results.append([i+1, GA,PA,ED])
            print(f"{i+1:04d},{GA:.03f},{PA:.03f},{ED:.03f}")
            print(f"{i+1:04d},{GA:.03f},{PA:.03f},{ED:.03f}", file=f_result)

        f_result.close()
