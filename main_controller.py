from experiment import Experiment



# datasets = {'iris', 'breast-cancer', 'iono', 'ecoli', 'wine'}
# Iris: N=150, d=4, k=3
if __name__ == '__main__':
    controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3)
    # controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3)
    # controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3)
    # controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3)
    # controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3)
    # controller = Experiment(dataset='iris', real_or_artificial='real-world', real_number_of_classes=3)

    controller.run_single_clusterings()
    controller.run_ensemling_clusterings()
    print('Lala')