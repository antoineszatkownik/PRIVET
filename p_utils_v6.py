import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import torch


CUTOFF = 1e-20
rdm=np.linspace(0,1,2) #for roc curve plot random baseline
threshold=-3

def load_data(path):
    f = open(path,"r")
    mat = []
    for line in f.readlines():
        if "YRI" in line:continue
        mat.append(list(map(int,list(line.rstrip("\n")))))
    mat = np.array(mat)
    f.close()
    return mat

def log_rank_in_cumulative(SIZE):
    p = 1. * np.arange(1, SIZE + 1) / SIZE
    log_p = np.log10(p)
    return log_p

def sorting(dist_NN):
    ind = np.argsort(dist_NN[0].numpy().reshape(-1,))
    p_NN_dist = dist_NN[0][ind]
    p_NN_idx = dist_NN[1][ind]

    return p_NN_dist, p_NN_idx


def cdf_extrapolate(x, A, alpha_slope): #used to project on the fit: cf \hat F(x) in paper
    px = 1-np.exp(-np.exp(A)*x**alpha_slope) #I didn't forgot to multiply by N, \hat A is in fact log(NA)
    #px = np.where(px>1, 1, px)
    return px

#this is the survival function of a binomial random var cf how \pi_i^{\rm ref} is computed in paper 
#ie "[...] ~(\ref{eq:proba_r}) now reads [...] " in paper
def pi_i_log_sorted(px, k, N):
    z=binom.sf(k, N, px, loc=0)
    if z<=CUTOFF:z=CUTOFF
    return z

#the estimation of intercept and slope can be achieved via maximum likelihood instead !!
#maximum likelihood estimation would allow to bypass the choice of a partition on which the fit is done
#ie bypass hyperparameter start_idx and end_idx which are chosen to be from 1% to 20% of datapoints by default
#the choice of the partition should be done by visually inspecting the tails of the Train-Train 1-NN distances CDF
#the fit should be done on the first mode if there are multiple ones (in that case 0.1%-5% is often appropriate)
def fit_nearest_neighbor_cdf(z_star, start_idx, end_idx):
    """
    Fit parameters A and alpha for the model:
    P(z_i^* < x | N) ≈ 1 - exp(-N * A * x^alpha)
    
    Args:
        z_star: Array of minimum distances (shape: [n_samples]).
        start_idx: Start index of the partition (inclusive).
        end_idx: End index of the partition (exclusive).
    
    Returns:
        A: Estimated amplitude parameter.
        alpha: Estimated power-law exponent.
        std_err_A: Standard error of A.
        std_err_alpha: Standard error of alpha.
    """
    # Sort distances and compute survival probability
    z_sorted = np.sort(z_star)
    N = len(z_sorted)
    
    # Survival probability: P(z_i^* >= x) = 1 - CDF(x)
    survival = 1 - (np.arange(N) + 1) / N  # 1 - (i+1)/N
    
    # Select partition and filter invalid points (survival > 0)
    survival_part = survival[start_idx:end_idx]
    z_part = z_sorted[start_idx:end_idx]
    #mask = survival_part > 0
    #survival_part = survival_part[mask]
    #z_part = z_part[mask]
    
    # Linearize: Y = ln(-ln(survival)), X = ln(z)
    Y = np.log(-np.log(survival_part))
    X = np.log(z_part)
    
    # Linear regression (Y = intercept + alpha * X)
    X_design = np.vstack([np.ones_like(X), X]).T
    beta, _, _, _ = np.linalg.lstsq(X_design, Y, rcond=None)
    intercept, alpha = beta[0], beta[1]
    
    # Compute A and its error
    #A = np.exp(intercept) / N  # intercept = ln(N * A) => A = e^(intercept)/N
    
    # Covariance matrix for error estimates
    residuals = Y - X_design @ beta
    mse = np.sum(residuals**2) / (len(Y) - 2)
    XTX_inv = np.linalg.inv(X_design.T @ X_design)
    cov_matrix = mse * XTX_inv
    std_err_intercept, std_err_alpha = np.sqrt(np.diag(cov_matrix))
    #std_err_A = (np.exp(intercept) / N) * std_err_intercept  # Error propagation

    X_bis = np.log(z_sorted)
    # Compute the standard error of prediction for each point
    sigma_Y_pred = np.sqrt(cov_matrix[0, 0] + 2 * X_bis * cov_matrix[0, 1] + cov_matrix[1, 1] * X_bis**2)

    
    return intercept, alpha, std_err_intercept, std_err_alpha, sigma_Y_pred


def gpu_nearest_neighbors(X, Y=None, k=1,
                           distance='hamming',
                           chunk_size=128,
                           device='cuda',
                           verbose=False):
    """
    Compute the k nearest neighbors for each sample in X. If Y is provided, for each sample in X
    find the k nearest neighbors among Y. When Y is None, find nearest neighbors within X (self-comparison
    with self-distance excluded).

    This function expect samples to be flattened to 1D vectors, ie X is a tensor of shape (N_samples, N_features)

    distances taken into account are: "hamming", "standard_euclidean" and "feature_normalized_euclidean"
    feature_normalized_euclidean is euclidean distance further divided by sqrt(N_features) 
    (feature_normalized_euclidean is used in 4.1 https://arxiv.org/abs/2301.13188)

    Returns:
        A tuple (dists, indices):
         - dists: Array of shape (n_samples_X, k) with nearest neighbor distances.
         - indices: Array of shape (n_samples_X, k) with indices of the neighbors (relative to Y, or X if Y is None).
    """
    # Device
    device = torch.device(device)
    X = X.to(device)
    same = Y is None
    Y = X if same else Y.to(device)

    nX, dim = X.shape
    nY = Y.shape[0]

    # For Euclidean distances
    if distance in ('standard_euclidean', 'feature_normalized_euclidean'):
        x_sq = (X.float() ** 2).sum(dim=1)
        y_sq = x_sq if same else (Y.float() ** 2).sum(dim=1)
        normalize = (distance == 'feature_normalized_euclidean')
        factor = dim if normalize else 1

    # Output buffers
    best_d = torch.full((nX, k), float('inf'), device=device)
    best_i = torch.full((nX, k), -1,          device=device, dtype=torch.long)

    outer = range(0, nX, chunk_size)
    if verbose:
        outer = tqdm(outer, desc='Rows')

    for i in outer:
        end_i = min(i + chunk_size, nX)
        Xc = X[i:end_i]
        if distance in ('standard_euclidean', 'feature_normalized_euclidean'):
            xn = x_sq[i:end_i]

        chunk_d = torch.full((end_i-i, k), float('inf'), device=device)
        chunk_i = torch.full((end_i-i, k), -1,        device=device, dtype=torch.long)

        inner = range(0, nY, chunk_size)
        if verbose and not same:
            inner = tqdm(inner, desc='Cols', leave=False)

        for j in inner:
            end_j = min(j + chunk_size, nY)
            Yc = Y[j:end_j]

            if distance == 'hamming':
                # exact integer mismatch counts
                d = (Xc.unsqueeze(1) != Yc.unsqueeze(0)) \
                        .to(torch.int32)                         \
                        .sum(dim=2)                              \
                        .to(torch.float32)
            else:
                yn = y_sq[j:end_j]
                xy = torch.mm(Xc.float(), Yc.float().t())
                sq = (xn[:, None] + yn[None, :] - 2*xy) / factor
                d = torch.sqrt(torch.clamp(sq, min=0.0))

            if same:
                # mask self-distances
                rows = torch.arange(i, end_i, device=device)
                cols = torch.arange(j, end_j, device=device)
                mask = rows.unsqueeze(1) == cols.unsqueeze(0)
                d.masked_fill_(mask, float('inf'))

            # merge into top-k
            concat_d = torch.cat([chunk_d, d], dim=1)
            idx_block = torch.arange(j, end_j, device=device).expand(end_i-i, -1)
            concat_i = torch.cat([chunk_i, idx_block], dim=1)

            chunk_d, pos = torch.topk(concat_d, k, largest=False, sorted=True)
            chunk_i = torch.gather(concat_i, 1, pos)

        best_d[i:end_i] = chunk_d
        best_i[i:end_i] = chunk_i

    return best_d.cpu(), best_i.cpu()



def generate_fake_synth(train: np.ndarray, synth: np.ndarray, indices: np.ndarray, f_fake: float, f_copy: float ) -> np.ndarray:
    """
    We suppose having N_train = N_synth, or more subtly, N is the number of samples supposed to be common to train and synthetic set

    Parameters:
    indices: Array of shape (n_samples_X, k) with indices of the neighbors (relative to Y, or X if Y is None).
    f_fake \in [0,1]: fraction of synthetic data containing leaked information from the training data
    f_copy \in [0,1]: amount of leaked information (along the SNPs)
    """

    N, L = synth.shape #N_samples, N_features
    n = int(np.ceil(N*f_fake)) #number of samples in synth

    fake = np.zeros((N,L))
    fake[n:] = synth[n:] # fake samples from n,n+1,...,N are copied from synth
    #fake samples from 0,...,n-1 are leaked synthetic
    for s in range(n):
        idx_train = indices[s] # idx of the real (train) sample that is 1-NN of that synthetic
        # Hybridize between synth and train
        snp_mask = np.zeros(L,dtype=bool)
        snp_mask[:int(f_copy*L)] = True
        np.random.shuffle(snp_mask) #is it necessary?
        fake[s] = np.where(snp_mask == True, train[idx_train], synth[s])

    return torch.tensor(fake,dtype=torch.uint8)#fake.astype(int)


def compute_scores(mat_syn_train, mat_syn_test, intercept, alpha, groundtruth_flag):

    N = mat_syn_train.shape[0] #size of d_{STr}^*

    #\delta \pi, \delta \pi train, \delta \pi test, px train, px test, rank train, rank test, dist train, dist test
    store_in_mat = np.zeros((N, 10))

    train_lookup = {int(row[1]): i for i, row in enumerate(mat_syn_train)}
    test_lookup = {int(row[1]): i for i, row in enumerate(mat_syn_test)}
    #pdb.set_trace()
    for idx_syn in range(N):
        rank_s_te = test_lookup.get(idx_syn)
        rank_s_tr = train_lookup.get(idx_syn)

        row_train = mat_syn_train[rank_s_tr]
        if groundtruth_flag:
            groundtruth_i = row_train[-1].item()
        dist_train = row_train[0].item()
    
        row_test = mat_syn_test[rank_s_te]
        dist_test = row_test[0].item()

        #pdb.set_trace()
    
        px_train = cdf_extrapolate(dist_train, intercept, alpha)
        px_test = cdf_extrapolate(dist_test, intercept, alpha)

        #pdb.set_trace()
    
        z_train_i=pi_i_log_sorted(px_train, rank_s_tr, N)
        z_test_i=pi_i_log_sorted(px_test, rank_s_te, N)

        #pdb.set_trace()
   
        delta_pi_i_new=np.log10(z_train_i)-np.log10(z_test_i)

        store_in_mat[idx_syn,0] = delta_pi_i_new
        store_in_mat[idx_syn,1] = z_train_i
        store_in_mat[idx_syn,2] = z_test_i
        store_in_mat[idx_syn,3] = px_train
        store_in_mat[idx_syn,4] = px_test
        store_in_mat[idx_syn,5] = rank_s_tr
        store_in_mat[idx_syn,6] = rank_s_te
        store_in_mat[idx_syn,7] = dist_train
        store_in_mat[idx_syn,8] = dist_test
        if groundtruth_flag:
            store_in_mat[idx_syn,9] = groundtruth_i

    #if groundtruth != None:
        #store_in_mat = np.hstack([store_in_mat,groundtruth.numpy().reshape(-1,1)])

    return store_in_mat

def PRIVET(train, test, synthetic, intercept, alpha, renormalization = None, distance='standard_euclidean', space="embedding", device=None, groundtruth = None):#device.type):

    ############################
    ## COMPUTE 1-NN distances ##
    ############################
    
    ## SYNTHETIC TO TRAIN
    #should be pytorch tensor here
    dist_NN_syn_tr = gpu_nearest_neighbors(synthetic, train, k=1, distance=distance,chunk_size=128,device=device,verbose=False)
    p_syn_tr_NN_dist, p_syn_tr_NN_idx = sorting(dist_NN_syn_tr)

    ## SYNTHETIC TO TEST
    dist_NN_syn_te = gpu_nearest_neighbors(synthetic, test, k=1,distance=distance,chunk_size=128,device=device,verbose=False)
    p_syn_te_NN_dist, p_syn_te_NN_idx = sorting(dist_NN_syn_te)

    #pdb.set_trace()

    tmp = dist_NN_syn_te[0] # for renormalization

    if renormalization != None:
        tmp = tmp / renormalization 
        p_syn_te_NN_dist = p_syn_te_NN_dist / renormalization #for plotting cdf
        if test.shape[0] > train.shape[0]:
            tmp = tmp * renormalization  
            p_syn_te_NN_dist = p_syn_te_NN_dist * renormalization #for plotting cdf


    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
    mat_syn_train = torch.concatenate([dist_NN_syn_tr[0], torch.arange(synthetic.shape[0]).reshape(-1,1), dist_NN_syn_tr[1]],axis=1)
    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
    mat_syn_test = torch.concatenate([tmp, torch.arange(synthetic.shape[0]).reshape(-1,1), dist_NN_syn_te[1]],axis=1)

    #pdb.set_trace()
    groundtruth_flag=False
    if groundtruth != None:
        #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
        mat_syn_train = torch.concatenate([dist_NN_syn_tr[0], torch.arange(synthetic.shape[0]).reshape(-1,1), dist_NN_syn_tr[1], groundtruth.reshape(-1,1)],axis=1)
        #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
        mat_syn_test = torch.concatenate([tmp, torch.arange(synthetic.shape[0]).reshape(-1,1), dist_NN_syn_te[1], groundtruth.reshape(-1,1)],axis=1)
        groundtruth_flag=True

    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on d_STr^*)
    sorted_mat_syn_train=mat_syn_train[mat_syn_train[:, 0].argsort()]
    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on d_STr^*)
    sorted_mat_syn_test = mat_syn_test[mat_syn_test[:, 0].argsort()]

    #pdb.set_trace()
    
    store_in_mat = compute_scores(sorted_mat_syn_train, sorted_mat_syn_test, intercept, alpha, groundtruth_flag=groundtruth_flag)
    
    return store_in_mat, p_syn_tr_NN_dist, p_syn_te_NN_dist

