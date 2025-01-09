import numpy as np
import pandas as pd
import scipy.special
from nltk.metrics.distance import edit_distance as calc_edit_distance

from settings import benchmark_settings

def get_accuracy(series_groundtruth, series_parsedlog):
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])

    correctly_grouped_events = 0
    correctly_grouped_templates = 0
    correctly_parsed_templates = 0

    for groundtruth_template, group in df_combined.groupby('groundtruth'):
        parsed_template = list(group['parsedlog'].unique())
        if len(parsed_template) == 1:
            if len(group) == series_parsedlog[series_parsedlog == parsed_template[0]].size:
                correctly_grouped_events += len(group)
                correctly_grouped_templates += 1
                if parsed_template[0] == groundtruth_template:
                    correctly_parsed_templates += 1

    GA = correctly_grouped_events / len(series_groundtruth)
    PGA = float(correctly_grouped_templates) / len(series_parsedlog_valuecounts)
    RGA = float(correctly_grouped_templates) / len(series_groundtruth_valuecounts)
    PTA = float(correctly_parsed_templates) / len(series_parsedlog_valuecounts)
    RTA = float(correctly_parsed_templates) / len(series_groundtruth_valuecounts)

    FGA = 0.0 if PGA == 0 and RGA == 0 else 2 * (PGA * RGA) / (PGA + RGA)
    FTA = 0.0 if PTA == 0 and RTA == 0 else 2 * (PTA * RTA) / (PTA + RTA)

    return GA, FGA, FTA, PGA, RGA, PTA, RTA

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
    rootdir = Path(__file__).absolute().parent.parent
    print(f"Project root: {rootdir}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--projects",
        type=str,
        default="Android,Apache,BGL,HDFS,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Spark,Thunderbird,Windows,Zookeeper"
        # default="Apache,BGL,HDFS,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Spark,Thunderbird,Zookeeper"
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    for project in args.projects.split(","):
        print(project)

        dataset_dir = rootdir / "2k_dataset"
        predic_file = Path(args.dirpath) / project / "predictions.csv"
        result_file = Path(args.dirpath) / project / "result.csv"

        df_parsedlog = pd.read_csv(predic_file, index_col=False, header='infer').map(str)
        df_groundtruth = pd.read_csv(dataset_dir/project/(project+"_2k.log_structured_corrected.csv"))
        series_parsedlog = df_parsedlog['EventTemplate']
        series_groundtruth = df_groundtruth['EventTemplate']

        f_result = open(result_file, "w")
        print("Batch,   GA,    PA,   FGA,   FTA,    ED")
        print("Batch,GA,PA,FGA,FTA,ED", file=f_result)

        edit_distance_in_batch = []
        for i in range(len(df_parsedlog)//batch_size):
            _series_parsedlog   = series_parsedlog.iloc[:(i+1)*batch_size].copy() #.str.replace( r"\s+", "", regex=True)
            _series_groundtruth = series_groundtruth.iloc[:(i+1)*batch_size].copy() #.str.replace( r"\s+", "", regex=True)

            GA, FGA, FTA, _,_,_,_ = get_accuracy(_series_groundtruth, _series_parsedlog)
            PA = get_PA(_series_groundtruth, _series_parsedlog)

            edit_distance,_ = get_ED(_series_groundtruth.tail(batch_size),
                                     _series_parsedlog.tail(batch_size))
            edit_distance_in_batch.append(edit_distance)
            ED = np.mean(edit_distance_in_batch)

            print(f"{i+1:04d},{GA:.04f},{PA:.04f},{FGA:.04f},{FTA:.04f},{ED:.04f}")
            print(f"{i+1:04d},{GA:.04f},{PA:.04f},{FGA:.04f},{FTA:.04f},{ED:.04f}", file=f_result)

        f_result.close()
