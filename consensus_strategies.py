from enum import Enum


class Consensus(Enum):
    AL = 'Average Linkage'
    CL = 'Complete Linkage'
    SL = 'Single Linkage'
    KM = 'K means'
    PAM = 'K medoids'
    SPC = 'Spectral'
    METIS = 'Metis'
    gSPEC = 'Spectral graph'
