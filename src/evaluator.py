import numpy as np
import pandas as pd
from nltk.metrics.distance import edit_distance

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
