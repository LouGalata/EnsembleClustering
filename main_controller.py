from experiment import Experiment



datasets = {'iris', 'breast-cancer', 'iono', 'ecoli', 'wine'}
# Iris: N=150, d=4, k=3
if __name__ == '__main__':

    # datasets = {('iris',3, 'real-world'),
    #             ('breast-cancer', 2, 'real-world'),
    #             ('iono',2, 'real-world'),
    #             ('ecoli',8, 'real-world'),
    #             ('wine',3, 'real-world'),
    #             ('banana', 2, 'artificial'),
    #             ('donut2', 2, 'artificial'),
    #             ('gaussians1', 2, 'artificial')}

    datasets = {('donut2', 2, 'artificial')}


    for dataset in datasets:
        for are_clusters_fixed in [True, False]:
            # for algorithm in ['SL', 'CL', 'AL', 'PAM', 'KM', 'SPC', 'METIS']:
            for algorithm in ['SL']:
                controller = Experiment(dataset=dataset[0], real_or_artificial=dataset[2], real_number_of_classes=dataset[1], are_clusters_fixed=are_clusters_fixed)

                controller.run_single_clusterings()
                # controller.create_ensembling_matrices()
                controller.run_ensemling_clusterings(algorithm)
