#v9 is  v6, except i'm now saving the idx_train and idx_test in final table output!! (TO DO: change v6 and all concerned script with this modif)


import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import pairwise_distances
#import seaborn as sns
from scipy.special import gammaln
from scipy.stats import binom
import torch
import pdb


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

def extract_nearest_neighbors(dij): #not used anymore
    """
    dij: matrix of pairwise distances train-synth ie contain TT, TS, ST, SS
    """
    N=dij.shape[0]//2
    vec_idx_NN_synth_to_train = np.argmin(dij[N:,:N],axis=1).reshape(1,-1) #nearest neighbor of synthetics in train so idx is for a training sample
    vec_dist_NN_synth_to_train = np.min(dij[N:,:N],axis=1).reshape(1,-1)

    return vec_dist_NN_synth_to_train, vec_idx_NN_synth_to_train

def cdf_extrapolate(x, A, alpha_slope): #used to project on the fit: cf \hat F(x) in paper
    px = 1-np.exp(-np.exp(A)*x**alpha_slope) #I didn't forgot to multiply by N, \hat A is in fact log(NA)
    #px = np.where(px>1, 1, px)
    return px

def cdf_extrapolate_gumbel(x, A, B): #used to project on the fit: cf \hat F(x) in paper
    px = 1-np.exp(-A*np.exp(B*x)) #I didn't forgot to multiply by N, \hat A is in fact log(NA)
    #px = np.where(px>1, 1, px)
    return px

#this is the survival function of a binomial random var cf how \pi_i^{\rm ref} is computed in paper 
#ie "[...] ~(\ref{eq:proba_r}) now reads [...] " in paper
def pi_i_log_sorted(px, k, N):
    z=binom.sf(k, N, px, loc=0)
    #z=np.where(z<CUTOFF, CUTOFF,z)
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


def compute_scores(mat_syn_train, intercept, alpha, groundtruth_flag):

    N = mat_syn_train.shape[0] #size of d_{STr}^*

    #\delta \pi, \delta \pi train, \delta \pi test, px train, px test, rank train, rank test, dist train, dist test
    store_in_mat = np.zeros((N, 7))

    train_lookup = {int(row[1]): i for i, row in enumerate(mat_syn_train)}
    #test_lookup = {int(row[1]): i for i, row in enumerate(mat_syn_test)}
    #pdb.set_trace()
    for idx_syn in range(N):
        #rank_s_te = test_lookup.get(idx_syn)
        rank_s_tr = train_lookup.get(idx_syn)

        row_train = mat_syn_train[rank_s_tr]
        if groundtruth_flag:
            groundtruth_i = row_train[-1].item()
        dist_train = row_train[0].item()
        idx_train = row_train[1].item() #this is in fact train
        idx_synth_tr = row_train[2].item() #this is synth
    
        #row_test = mat_syn_test[rank_s_te]
        #idx_synth_te = row_test[1].item()
        #dist_test = row_test[0].item()
        #idx_test = row_test[2].item()

        #print(idx_synth_tr == idx_synth_te)

        #pdb.set_trace()
    
        px_train = cdf_extrapolate(dist_train, intercept, alpha)
        #px_test = cdf_extrapolate(dist_test, intercept, alpha)

        #pdb.set_trace()
    
        z_train_i=pi_i_log_sorted(px_train, rank_s_tr, N)
        #z_test_i=pi_i_log_sorted(px_test, rank_s_te, N)

        #pdb.set_trace()
   
        #delta_pi_i_new=np.log10(z_train_i)-np.log10(z_test_i)

        #store_in_mat[idx_syn,0] = delta_pi_i_new
        store_in_mat[idx_syn,0] = z_train_i
        #store_in_mat[idx_syn,2] = z_test_i
        store_in_mat[idx_syn,1] = px_train
        #store_in_mat[idx_syn,4] = px_test
        store_in_mat[idx_syn,2] = rank_s_tr
        #store_in_mat[idx_syn,6] = rank_s_te
        store_in_mat[idx_syn,3] = dist_train
        #store_in_mat[idx_syn,8] = dist_test
        store_in_mat[idx_syn,4] = idx_synth_tr
        store_in_mat[idx_syn,5] = idx_train
        #store_in_mat[idx_syn,11] = idx_test
        if groundtruth_flag:
            store_in_mat[idx_syn,6] = groundtruth_i

    #if groundtruth != None:
        #store_in_mat = np.hstack([store_in_mat,groundtruth.numpy().reshape(-1,1)])

    return store_in_mat

def PRIVET(train, synthetic, intercept, alpha, renormalization = None, distance='standard_euclidean', space="embedding", device=None, groundtruth = None):#device.type):

    ############################
    ## COMPUTE 1-NN distances ##
    ############################
    
    ## SYNTHETIC TO TRAIN
    #should be pytorch tensor here
    dist_NN_syn_tr = gpu_nearest_neighbors(train, synthetic, k=1, distance=distance,chunk_size=128,device=device,verbose=False)
    p_syn_tr_NN_dist, p_syn_tr_NN_idx = sorting(dist_NN_syn_tr)

    ## SYNTHETIC TO TEST
    #dist_NN_syn_te = gpu_nearest_neighbors(test, synthetic, k=1,distance=distance,chunk_size=128,device=device,verbose=False)
    #p_syn_te_NN_dist, p_syn_te_NN_idx = sorting(dist_NN_syn_te)

    #pdb.set_trace()

    tmp = dist_NN_syn_tr[0] # for renormalization

    if renormalization != None:
        tmp = tmp / renormalization 
        p_syn_tr_NN_dist = p_syn_tr_NN_dist / renormalization #for plotting cdf
        if synthetic.shape[0] > train.shape[0]:
            tmp = tmp * renormalization  
            p_syn_tr_NN_dist = p_syn_tr_NN_dist * renormalization #for plotting cdf


    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
    mat_syn_train = torch.concatenate([tmp, torch.arange(train.shape[0]).reshape(-1,1), dist_NN_syn_tr[1]],axis=1)
    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
    #mat_syn_test = torch.concatenate([tmp, torch.arange(train.shape[0]).reshape(-1,1), dist_NN_syn_te[1]],axis=1)

    #pdb.set_trace()
    groundtruth_flag=False
    if groundtruth != None:
        #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
        mat_syn_train = torch.concatenate([tmp, torch.arange(train.shape[0]).reshape(-1,1), dist_NN_syn_tr[1], groundtruth.reshape(-1,1)],axis=1)
        #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
        #mat_syn_test = torch.concatenate([tmp, torch.arange(train.shape[0]).reshape(-1,1), dist_NN_syn_te[1], groundtruth.reshape(-1,1)],axis=1)
        #groundtruth_flag=True

    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on d_STr^*)
    sorted_mat_syn_train=mat_syn_train[mat_syn_train[:, 0].argsort()]
    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on d_STr^*)
    #sorted_mat_syn_test = mat_syn_test[mat_syn_test[:, 0].argsort()]

    #pdb.set_trace()
    
    store_in_mat = compute_scores(sorted_mat_syn_train, intercept, alpha, groundtruth_flag=groundtruth_flag)
    
    return store_in_mat, p_syn_tr_NN_dist

    
def plot(flist, delta_pi, p_TrTr_NN, sorted_mat_train_fake, sorted_mat_test_fake, K_train, mu_train, sigma_train, A_train, alpha_train, err_gaussian_train, err_power_law_train):
    N=delta_pi.shape[0]
    
    x_fpr, y_tpr, auc_roc, y_precision, x_recall, auc_pr = roc(flist,delta_pi)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4), dpi=200, facecolor="white")
    ax[0].plot(x_fpr,y_tpr,label=f"AUC={np.round(auc_roc,2)}",marker=".",markersize=1,linewidth=1)
    ax[0].plot(rdm,rdm,color="black",linestyle="dashed")
    ax[0].set_xlim([-0.05,1.05])
    ax[0].set_ylim([-0.05,1.05])
    ax[0].set_title("ROC")
    ax[0].set_xlabel(r"False positive rate ($\frac{fp}{fp + tn}$)")
    ax[0].set_ylabel(r"True positive rate ($\frac{tp}{tp + fn}$)")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(x_recall,y_precision,label=f"AUC={np.round(auc_pr,2)}",marker=".",markersize=1,linewidth=1)
    ax[1].set_xlim([-0.05,1.05])
    ax[1].set_ylim([-0.05,1.05])
    ax[1].set_title("PR")
    ax[1].set_ylabel(r"Precision($\frac{tp}{tp + fp}$)")
    ax[1].set_xlabel(r"Recall ($\frac{tp}{tp + fn}$)")
    ax[1].legend()
    ax[1].grid()

    p = 1. * np.arange(1, N + 1) / N
    log_p = np.log10(p)

    ax[2].scatter(p_TrTr_NN, log_p,color="olive",label=r"$d^*_{TrTr}$",marker="x",s=1)

    if err_gaussian_train< err_power_law_train:
        Y_pred_gauss_Tr = K_train * np.exp(-0.5 * (p_TrTr_NN - mu_train)**2 / sigma_train**2)
        ax[2].plot(p_TrTr_NN, np.log10(Y_pred_gauss_Tr),color='olive',label="gaussian fit (Tr)",linestyle="dashed",alpha=.7)

    if err_power_law_train<err_gaussian_train:
        Y_pred_powerlaw_Tr = A_train + alpha_train*np.log(p_TrTr_NN)
        ax[2].plot(p_TrTr_NN, Y_pred_powerlaw_Tr/np.log(10),color='darkgreen',label="powerlaw fit (Tr)",linestyle="dashed",alpha=.7)

    ax[2].scatter(sorted_mat_train_fake[:,0], log_p,color="deeppink", label=r"$d_{STr}^*$",marker="x",s=1)
    ax[2].scatter(sorted_mat_test_fake[:,0], log_p,color="rebeccapurple", label=r"$d_{STe}^*$",marker="x",s=1)

    ax[2].set_xlabel('$d_{ij}$')
    ax[2].set_ylabel('$(log10) F_{d_{ij}}$')

    unit=1/N
    log_unit=np.log10(unit)

    ax[2].hlines(log_unit,min(sorted_mat_train_fake[:,0]),max(sorted_mat_train_fake[:,0]),color='black',linestyles="dashed",alpha=.2)
    ax[2].set_ylim([log_unit-0.5,0.2]) #N=1668 -> -8, N=66 -> -4.4
    ax[2].set_title(fr"N={N}, $f_{{\mathrm{{fake}}}}={f_fake}, f_{{\mathrm{{copy}}}}={f_copy}$")
    ax[2].grid(True)
    ax[2].legend(markerscale=6)

    fig.tight_layout()

    plt.show()

def plot_heatmap(data, title,fake_range, copy_range, cmap, file_name):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(data.T, cmap=cmap)

    # Set a limited number of ticks
    ax.set_xticks(np.linspace(0, len(fake_range) - 1, 10))  # 10 x-ticks
    ax.set_xticklabels(np.round(np.linspace(fake_range[0], fake_range[-1], 10), 3))

    ax.set_yticks(np.linspace(0, len(copy_range) - 1, 10))  # 10 y-ticks
    ax.set_yticklabels(np.round(np.linspace(copy_range[0], copy_range[-1], 10), 3))

    plt.xlabel(r"$f_{fake}$")
    plt.ylabel(r"$f_{copy}$")
    plt.title(title)
    ax.invert_yaxis()  # Ensure higher values are at the top
    plt.show()

def write_file(text):
    f=open('65K_seed42_modified_cumulative.txt','a') 
    f.write(text+"\n")
    f.close()
    f=open('65K_seed42_modified_cumulative.txt','a')

def get_predictions(pred, groundtruth):
    tp = (pred*groundtruth).sum()
    fn = ((~pred)*groundtruth).sum()
    fp = (pred*(~groundtruth)).sum()
    tn = ((~pred)*(~groundtruth)).sum()

    return tp.item(), fn.item(), fp.item(), tn.item()

styles = [{'color': 'olive', 'label': r'$d^*_{TrTr}$', 'marker': 'x', 's': 1},
          {'color': 'deeppink', 'label': r'$d^*_{STr}$', 'marker': 'x', 's': 1},
          {'color': 'deepskyblue', 'label': r'$d^*_{STe}$', 'marker': 'x', 's': 1}]

def plot_CDFs(p_tr_tr, p_syn_tr, p_syn_te, styles):

    FONTSIZE = 13
    
    plt.rcParams.update({
        'axes.labelsize': FONTSIZE,
        'axes.titlesize': FONTSIZE,
        'xtick.labelsize': FONTSIZE,
        'ytick.labelsize': FONTSIZE
    })

    # Define style parameters
    real_style = styles[0]
    synth_style = styles[1]
    te_synth_style = styles[2]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
    
    log_p_real = log_rank_in_cumulative(p_tr_tr.shape[0])
    log_p_synth = log_rank_in_cumulative(p_syn_tr.shape[0])

    ax.scatter(p_tr_tr, log_p_real, **real_style)
    ax.scatter(p_syn_tr, log_p_synth, **synth_style)
    ax.scatter(p_syn_te, log_p_synth, **te_synth_style)

    ax.set_xlabel('$(\log_{10})d_{ij}$')
    ax.set_ylabel('$(\log_{10}) F_{d_{ij}}$')
    ax.set_xscale('log')
    ax.grid(True)
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.2), markerscale=10,fontsize=FONTSIZE)
    return fig, ax, FONTSIZE