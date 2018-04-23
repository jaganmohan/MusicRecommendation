# MusicRecommendation
content based recommendation system 
ALS.py : Takes playcount matrix as input and factorizes into user latent vectors and item latent vectors of size 50
WMF.py : Takes playcount matrix as input. Two matrices are then created preference and confidence matrix. Then the matrix is factorized using the two matrices. This way of factorizing is achieved by weighted distribution of the input.
WMF is proposed by Hu et al. in their paper "Collaborative Filtering for Implicit Feedback Datasets"
TrainMFCC.py: takes MFCC's as input to the CNN network and trained against the item latent factors obtained from the above steps.
TrainMelspec.py: takes Mel spectrograms as input to the CNN network and trained against the item latent factors obtained from the above steps.
