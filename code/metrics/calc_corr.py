from .metrics import *
from .dl_metrics import *

def calc_corr(data, method, random_seed):
    features_list = list(data.columns)[:-1]
    label = list(data.columns)[-1]
    corr_dict = {}

    if method == 'dstar':
        for feature in features_list:
            corr_dict[feature] = dstar(data[feature], data[label])
    elif method == 'ochiai':
        for feature in features_list:
            corr_dict[feature] = ochiai(data[feature], data[label])
    elif method == 'barinel':
        for feature in features_list:
            corr_dict[feature] = barinel(data[feature], data[label])
    elif method == "ER1":
        for feature in features_list:
            corr_dict[feature] = ER1(data[feature], data[label])
    elif method == "ER5":
        for feature in features_list:
            corr_dict[feature] = ER5(data[feature], data[label])
    elif method == "GP02":
        for feature in features_list:
            corr_dict[feature] = GP02(data[feature], data[label])
    elif method == "GP03":
        for feature in features_list:
            corr_dict[feature] = GP03(data[feature], data[label])
    elif method == "GP19":
        for feature in features_list:
            corr_dict[feature] = GP19(data[feature], data[label])
    elif method == "Jaccard":
        for feature in features_list:
            corr_dict[feature] = Jaccard(data[feature], data[label])
    elif method == "Op2":
        for feature in features_list:
            corr_dict[feature] = Op2(data[feature], data[label])

    elif method == "MLP-FL":
        corr_dict = MLPBaseline(data[features_list], data[label], random_seed)
    elif method == "MLP-FLS":
        corr_dict = MLPSimplified(data[features_list], data[label], random_seed)
    elif method == "CNN-FL":
        corr_dict = CNNBaseline(data[features_list], data[label], random_seed)
    elif method == "CNN-FLS":
        corr_dict = CNNSimplified(data[features_list], data[label], random_seed)
    elif method == "RNN-FL":
        corr_dict = RNNBaseline(data[features_list], data[label], random_seed)
    elif method == "RNN-FLS":
        corr_dict = RNNSimplified(data[features_list], data[label], random_seed)

    else:
        raise Exception(f"Argument value error: No method '{method}'")
    return corr_dict
