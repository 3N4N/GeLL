import numpy as np
import pandas as pd
import scipy.special
from nltk.metrics.distance import edit_distance
from sklearn.metrics import accuracy_score

from settings import benchmark_settings

pd.options.display.float_format = "{:,.2f}".format

def get_GA(group, ground_list):
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

def get_ED(predictions, groundtruths):

    df = pd.DataFrame({'predictions': predictions, 'groundtruths': groundtruths})
    df['predictions_']  = df["predictions"].str.replace( r"\s+", "", regex=True)
    df['groundtruths_'] = df["groundtruths"].str.replace(r"\s+", "", regex=True)
    edit_distance_result = []
    for i, j in zip(
        np.array(df['predictions_'].values, dtype="str"),
        np.array(df['groundtruths_'].values, dtype="str"),
    ):
        edit_distance_result.append(edit_distance(i, j))

    edit_distance_result_mean = np.mean(edit_distance_result)
    edit_distance_result_std  = np.std(edit_distance_result)

    return edit_distance_result_mean, edit_distance_result_std



def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (
            parsed_eventId,
            series_groundtruth_logId_valuecounts.index.tolist(),
        )
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if (
                logIds.size
                == series_groundtruth[series_groundtruth == groundtruth_eventId].size
            ):
                accurate_events += logIds.size
                error = False
        if error and debug:
            print(
                "(parsed_eventId, groundtruth_eventId) =",
                error_eventIds,
                "failed",
                logIds.size,
                "messages",
            )
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)
    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy

def evaluate(df_parsedlog):
    df_parsedlog["Predict_NoSpaces"] = df_parsedlog["Predict"].str.replace(
        r"\s+", "", regex=True
    )
    df_parsedlog["EventTemplate_NoSpaces"] = df_parsedlog["EventTemplate"].str.replace(
        r"\s+", "", regex=True
    )
    accuracy_exact_string_matching = accuracy_score(
        np.array(df_parsedlog.EventTemplate_NoSpaces.values, dtype="str"),
        np.array(df_parsedlog.Predict_NoSpaces.values, dtype="str"),
    )
    edit_distance_result = []
    for i, j in zip(
        np.array(df_parsedlog.EventTemplate_NoSpaces.values, dtype="str"),
        np.array(df_parsedlog.Predict_NoSpaces.values, dtype="str"),
    ):
        edit_distance_result.append(edit_distance(i, j))

    edit_distance_result_mean = np.mean(edit_distance_result)
    edit_distance_result_std = np.std(edit_distance_result)
    (precision, recall, f_measure, accuracy_GA) = get_accuracy(
        df_parsedlog["EventTemplate_NoSpaces"], df_parsedlog["Predict_NoSpaces"]
    )
    return (
        accuracy_GA,
        accuracy_exact_string_matching,
        edit_distance_result_mean,
        edit_distance_result_std,
    )

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
        for i in range(len(df_parsedlog)//batch_size):
            df = df_parsedlog.iloc[:(i+1)*batch_size].copy()
            GA, PA, ED, _ = evaluate(df)
            results.append([i+1, GA,PA,ED])
            print(f"{i+1:04d},{GA:.03f},{PA:.03f},{ED:.03f}")
            print(f"{i+1:04d},{GA:.03f},{PA:.03f},{ED:.03f}", file=f_result)

        f_result.close()
