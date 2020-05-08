import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as clr

if __name__ == "__main__":
    dataset_name = 'iris'
    fixed_or_random = 'fixed_k'  # random_k
    consensus_algorithms = ['AL', 'SL', 'AL', 'Kmeans', 'Kmedoids', 'Spectral']
    bars = ['BA', 'CO', 'TMB', 'FCM', 'WCT', 'WTQ']
    path = os.path.join('out', dataset_name, fixed_or_random)

    scores = [score for score in os.listdir(path) if score.endswith('scores.csv') and score.startswith('Metis')]

    score_dict = dict()
    for score in scores:
        consensus = score.split('_')[0]
        score_dict[consensus] = pd.read_csv(path + '/' + score, sep='\t').to_dict()

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

    cmap = clr.LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#4f5fff'),
                                                  (1, '#efefff')], N=256)

    pd.DataFrame(acc_deformed).plot(kind='bar', rot=0, colormap=cmap, grid=False)
    plt.title(dataset_name + ' ' + 'Accuracy')
    plt.savefig(path + '/total_accuracy.png')
    plt.close()

    pd.DataFrame(jac_deformed).plot(kind='bar', rot=0, colormap=cmap, grid=False)
    plt.title(dataset_name + ' ' + 'Jaccard')
    plt.savefig(path + '/total_jaccard.png')
    plt.close()

    pd.DataFrame(nmi_deformed).plot(kind='bar', rot=0, colormap=cmap, grid=False)
    plt.title(dataset_name + ' ' + 'NMI')
    plt.savefig(path + '/total_nmi.png')
    plt.close()
