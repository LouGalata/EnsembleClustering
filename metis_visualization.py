import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as clr


def get_reshaped_scores(score_dict, bars):
    # Make Jaccardi Plot
    jaccard_dict = dict()
    for key, _ in score_dict.items():
        jaccard_dict[key] = score_dict[key]['jaccard']

    # NMI Plot
    nmi_dict = dict()
    for key, values in score_dict.items():
        nmi_dict[key] = score_dict[key]['nmi']

    # Accuracy Plot
    acc_dict = dict()
    for key, values in score_dict.items():
        acc_dict[key] = score_dict[key]['accuracy']

    acc_deformed = dict()
    for cnt, bar in enumerate(bars):
        acc_deformed_in = dict()
        for key, value in acc_dict.items():
            acc_deformed_in[key] = value[cnt]
        acc_deformed[bar] = acc_deformed_in

    jac_deformed = dict()
    for cnt, bar in enumerate(bars):
        jac_deformed_in = dict()
        for key, value in jaccard_dict.items():
            jac_deformed_in[key] = value[cnt]
        jac_deformed[bar] = jac_deformed_in

    nmi_deformed = dict()
    for cnt, bar in enumerate(bars):
        nmi_deformed_in = dict()
        for key, value in nmi_dict.items():
            nmi_deformed_in[key] = value[cnt]
        nmi_deformed[bar] = nmi_deformed_in

    return acc_deformed, jac_deformed, nmi_deformed


if __name__ == "__main__":
    dataset_name = 'donut2'
    # fixed_or_random = 'random_k'   # random_k, fixed_k
    consensus_algorithms = ['AL', 'SL', 'AL', 'Kmeans', 'Kmedoids', 'Spectral', 'Metis']
    # bars = ['BA', 'CO', 'TMB', 'FCM', 'WCT', 'WTQ']
    bars = ['CO']
    path = os.path.join('out', dataset_name)
    path_fixed = os.path.join('out', dataset_name, 'fixed_k')
    path_random = os.path.join('out', dataset_name, 'random_k')

    scores_fixed = [score for score in os.listdir(path_fixed) if score.endswith('scores.csv')]
    scores_random = [score for score in os.listdir(path_random) if score.endswith('scores.csv')]

    score_dict_fixed = dict()
    for score in scores_fixed:
        consensus = score.split('_')[0]
        score_dict_fixed[consensus] = pd.read_csv(path_fixed + '/' + score, sep='\t').to_dict()

    score_dict_random = dict()
    for score in scores_random:
        consensus = score.split('_')[0]
        score_dict_random[consensus] = pd.read_csv(path_random + '/' + score, sep='\t').to_dict()

    acc_deformed, jac_deformed, nmi_deformed = get_reshaped_scores(score_dict_fixed, bars)
    acc_deformed_r, jac_deformed_r, nmi_deformed_r = get_reshaped_scores(score_dict_random, bars)

    acc_combined = dict()
    acc_combined['Fixed-k'] = acc_deformed['CO']
    acc_combined['Random-k'] = acc_deformed_r['CO']

    jac_combined = dict()
    jac_combined['Fixed-k'] = jac_deformed['CO']
    jac_combined['Random-k'] = jac_deformed_r['CO']

    nmi_combined = dict()
    nmi_combined['Fixed-k'] = nmi_deformed['CO']
    nmi_combined['Random-k'] = nmi_deformed_r['CO']

    cmap = clr.LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#289ef3'),
                                                  (1, '#d1e1e9')], N=256)

    pd.DataFrame(acc_combined).plot(kind='bar', rot=0, colormap=cmap, grid=False)
    plt.title(dataset_name + ' ' + 'Accuracy')
    plt.savefig(path + '/metis_total_accuracy.png')
    plt.close()

    pd.DataFrame(jac_combined).plot(kind='bar', rot=0, colormap=cmap, grid=False)
    plt.title(dataset_name + ' ' + 'Jaccard')
    plt.savefig(path + '/metis_total_jaccard.png')
    plt.close()

    pd.DataFrame(nmi_combined).plot(kind='bar', rot=0, colormap=cmap, grid=False)
    plt.title(dataset_name + ' ' + 'NMI')
    plt.savefig(path + '/metis_total_nmi.png')
    plt.close()
