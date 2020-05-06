from experiment import Experiment



# datasets = {'iris', 'breast-cancer', 'iono', 'ecoli', 'wine'}
# Iris: N=150, d=4, k=3
if __name__ == '__main__':
    for are_clusters_fixed in [True, False]:
        for algorithm in ['SL', 'CL', 'AL', 'PAM', 'KM', 'SPC']:
            controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3, are_clusters_fixed=are_clusters_fixed)
            # controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3)
            # controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3)
            # controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3)
            # controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3)
            # controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3)

            controller.run_single_clusterings()
            controller.create_ensembling_matrices()
            controller.run_ensemling_clusterings(algorithm)
