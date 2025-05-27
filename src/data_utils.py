import numpy as np
import torch #TODO: remove

#for loading genetic data, specifically the 66 samples datasets
def load_data(path):
    f = open(path,"r")
    mat = []
    for line in f.readlines():
        if "YRI" in line:continue
        mat.append(list(map(int,list(line.rstrip("\n")))))
    mat = np.array(mat)
    f.close()
    return mat

#For controlled experiments on genetic data: it creates the pseudo-synthetic data
def generate_fake_synth(train: np.ndarray, synth: np.ndarray, indices: np.ndarray, f_fake: float, f_copy: float ) -> np.ndarray:
    """
    We suppose having N_train = N_synth, or more subtly, N is the number of samples supposed to be common to train and synthetic set

    Parameters:
    indices: Array of shape (n_samples_X, k) with indices of the neighbors (relative to Y, or X if Y is None).
    f_fake \in [0,1]: fraction of synthetic data containing leaked information from the training data
    f_copy \in [0,1]: amount of leaked information (along the SNPs)
    """

    N, L = synth.shape #N_samples = N_synth, N_features
    n = int(np.ceil(N*f_fake)) #number of synthetic samples that are leaked

    fake = np.zeros((N,L)) #this is the pseudo-synthetic data
    fake[n:] = synth[n:] # fake samples indexed from n,n+1,...,N are copied from synth
    #fake samples from 0,...,n-1 are leaked synthetic
    for s in range(n):
        idx_train = indices[s] # idx of the real (train) sample that is 1-NN of that synthetic
        # Hybridize between synth and train
        snp_mask = np.zeros(L,dtype=bool)
        snp_mask[:int(f_copy*L)] = True
        np.random.shuffle(snp_mask) #is it necessary?
        fake[s] = np.where(snp_mask == True, train[idx_train], synth[s])

    return torch.tensor(fake,dtype=torch.uint8) #TODO: leave it as np array