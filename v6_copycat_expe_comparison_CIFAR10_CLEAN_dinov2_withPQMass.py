import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import sys
import os
import torchvision.datasets as datasets
import io
from PIL import Image
import time
# caution: path[0] is reserved for script path (or '' in REPL)
ENV_PATH = "your/path/to/dgm-eval/folder"
sys.path.insert(1, ENV_PATH)

ROOT_PATH = "your/path/to/root"
sys.path.append(ROOT_PATH)  # Replace with your actual path

CACHE_TORCH_PATH = "your/path/to/.cache/torch"
os.environ["TORCH_HOME"] = CACHE_TORCH_PATH

from p_utils_v6 import *


ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

import os
import random
import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder
from collections import defaultdict
from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fld.metrics.FLD import FLD  # if you use FLD later

from fld.metrics.AuthPct import AuthPct
from fld.metrics.CTTest import CTTest

from pqm import pqm_pvalue, pqm_chi2


# Downsample synthetic dataset: select 1000 images per class
def downsample_dataset(dataset, n_per_class=1000):
    class_counts = defaultdict(int)
    selected_indices = []
    # dataset.samples is a list of (filepath, class) pairs for ImageFolder.
    for idx, (_, label) in enumerate(dataset.samples):
        if class_counts[label] < n_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
    return torch.utils.data.Subset(dataset, selected_indices)

def build_mixed_feature_set_determinist(gen_feat, train_feat, beta=0.0):
    """
    Create a combined feature set where the first `alpha` fraction are training features 
    and the remaining are synthetic features (deterministic order).
    
    Parameters:
        gen_feat (Tensor): Features from the synthetic dataset (shape: [N, D]).
        train_feat (Tensor): Features from the training dataset (shape: [M, D]).
        alpha (float): Fraction of the output set to take from training data.
    
    Returns:
        Tensor: Combined feature set of size equal to gen_feat, with training features first.
    """
    N = gen_feat.shape[0]
    n_real = int(beta * N)
    n_syn = N - n_real

    # Take the first n_real training samples and first n_syn synthetic samples
    real_feats = train_feat[:n_real]  # Deterministic: first n_real samples
    syn_feats = gen_feat[:n_syn]      # Deterministic: first n_syn samples
    
    # Concatenate (training features first, synthetic features after)
    combined = torch.cat([real_feats, syn_feats], dim=0)
    return combined  # No shuffling


class JPEGQuality(object):
    def __init__(self, quality=75):
        """
        Args:
            quality (int): JPEG quality, 1 (worst) to 95 (best). 75 is a good default.
        """
        assert 1 <= quality <= 95, "quality must be between 1 and 95"
        self.quality = quality

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Re-encoded JPEG at self.quality.
        """
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)

jpg75 = T.Compose([
    JPEGQuality(quality=75),
])

crop28 = T.Compose([
    T.CenterCrop(28),
    T.Pad(2),
])

class Posterize(object):
    def __init__(self, bits=5):
        self.bits = bits
    def __call__(self, img):
        return T.functional.posterize(img, self.bits)

posterize = T.Compose([
    Posterize(bits=5),
])

elastic = T.Compose([
    T.ElasticTransform(),
])

transforms_lst = [posterize, crop28, jpg75, elastic] #jpg75

train_dataset = CIFAR10(root="data", train=True, download=True)
test_dataset = CIFAR10(root="data", train=False, download=True)

feature_extractor = DINOv2FeatureExtractor()

train_feat = feature_extractor.get_features(train_dataset, name="train", recompute=False)
test_feat = feature_extractor.get_features(test_dataset, name="train", recompute=False)

############################
## COMPUTE 1-NN distances ##
############################
#Train-Train
dist_NN_tr_tr = gpu_nearest_neighbors(train_feat, k=1, distance='standard_euclidean',chunk_size=128,device=device.type,verbose=False)
p_tr_tr_NN_dist, p_tr_tr_NN_idx = sorting(dist_NN_tr_tr)

############################
## FIT CDF on Train-Train ##
############################

N_train = train_feat.shape[0]
partition_start = 0.001 #this is a hyperparameter of PRIVET
partition_end = 0.02 #this is a hyperparameter of PRIVET
#default would be to fit on 1% to 20%. but depending on shape of CDF (if there are multiple modes), 
#we want to fit the first mode, because we are interested in lower tail
#to be 100% sure on the partition of the fit, visual inspection of the CDF of 1-NN between train-train is recommended

start = int(np.ceil(partition_start*N_train)) #int(0.01*N) # if N is small start = 0 --> problem with log
end = int(partition_end*N_train)

# Fit parameters (adjust start/end indices to avoid extremes)
intercept, alpha, std_err_intercept, std_err_alpha, sigma_Y_pred = fit_nearest_neighbor_cdf(p_tr_tr_NN_dist.numpy().reshape(-1,), start_idx=start, end_idx=end)

print(f"Estimated intercept = {intercept:.2f} ± {std_err_intercept:.2f}")
print(f"Estimated alpha = {alpha:.2f} ± {std_err_alpha:.2f}")

transforms_lst = [posterize, crop28, jpg75, elastic]
transform_names = ['Posterize', 'Center crop 28', 'JPG 75', 'Elastic transform']


# metrics storage
beta_values = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
metrics = {
    name: {
        'NPL':   [],
        'FLD':   [],
        'Auth':  [],
        'CT':    [],
        'PQM':   []
    } for name in transform_names
}

time_PRIVET, time_FLD, time_AUTH, time_CT, time_PQM = [], [], [], [], []

SYNTH_DATA_PATH = "your/path/to/CIFAR10-PFGMPP"

torch.manual_seed(42)
for transform_fn, name in zip(transforms_lst, transform_names):
    print(name)
    # extract features once per transform
    train_dataset_trans = CIFAR10(root="data", train=True, download=True, transform=transform_fn)
    train_feat_trans = feature_extractor.get_features(train_dataset_trans, name=f"{name}_train", recompute=False)

    synthetic_full = ImageFolder(root=SYNTH_DATA_PATH, transform=transform_fn)
    synthetic_dataset = downsample_dataset(synthetic_full, n_per_class=1000)
    gen_feat = feature_extractor.get_features(synthetic_dataset, name=f"{name}_syn", recompute=False)

    # shuffle train features to mix in
    shuffled_indices = torch.randperm(train_feat_trans.shape[0], generator=torch.Generator().manual_seed(42))
    train_feat_shuf = train_feat_trans[shuffled_indices]

    shuffled_indices = torch.randperm(gen_feat.shape[0], generator=torch.Generator().manual_seed(42))
    gen_feat = gen_feat[shuffled_indices]

    for beta in beta_values:
        print(beta)
        mixed_feat = build_mixed_feature_set_determinist(
            gen_feat, train_feat_shuf, beta=beta
        )

        start = time.time()
        # PRIVET → NPL
        store, p_tr, p_te = PRIVET(
            train_feat, test_feat, mixed_feat,
            intercept, alpha,
            renormalization=5**(1/25),
            distance='standard_euclidean',
            device=device.type
        )
        end = time.time()
        time_PRIVET.append(end-start)
        NPL = (store[:,0] <= threshold).sum()
        metrics[name]['NPL'].append(NPL)

        start = time.time()
        # FLD generalization gap
        fld_gap = FLD(eval_feat="gap").compute_metric(train_feat, test_feat, mixed_feat)
        metrics[name]['FLD'].append(fld_gap)
        end = time.time()
        time_FLD.append(end-start)

        start = time.time()
        # AuthPct
        # code from https://github.com/marcojira/fld/blob/main/fld/metrics/AuthPct.py was changed at line 41
        # return (100 * torch.sum(authen) / len(authen)).item() - 50
        # became
        # return (100 * torch.sum(authen) / len(authen)).item() - 50, authen
        # authen is a list of booleans, one per synthetic, True if sample is deemed authentic, False otherwise
        auth_mask = AuthPct().compute_metric(train_feat, test_feat, mixed_feat)[1]
        auth_score = (~auth_mask.detach().cpu().numpy()).sum()
        metrics[name]['Auth'].append(auth_score)
        end = time.time()
        time_AUTH.append(end-start)

        start = time.time()
        # C_T
        ct_val = CTTest().compute_metric(train_feat, test_feat, mixed_feat)
        metrics[name]['CT'].append(ct_val)
        end = time.time()
        time_CT.append(end-start)

        #PQMass
        start = time.time()
        chi2_stat_synth_tr = pqm_chi2(mixed_feat, train_feat, re_tessellation = 1000)
        chi2_stat_synth_te = pqm_chi2(mixed_feat, test_feat, re_tessellation = 1000)
        end = time.time()
        time_PQM.append(end-start)
        chi2_stat_synth_tr = np.array(chi2_stat_synth_tr)
        chi2_stat_synth_te = np.array(chi2_stat_synth_te)
        pqm_chi2_gap = pqm_chi2_gap = np.nanmean(chi2_stat_synth_te) - np.nanmean(chi2_stat_synth_tr)#chi2_stat_synth_te.mean() - chi2_stat_synth_tr.mean()
        metrics[name]['PQM'].append(pqm_chi2_gap)

import pickle
with open('v6_copycat_expe_comparison_CIFAR10_CLEAN_dinov2_withPQMass_METRICS_DIC.pickle', 'wb') as handle:
    pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


print(f"time elapsed PRIVET mean = {np.mean(time_PRIVET)}, std = {np.std(time_PRIVET)}")
print(f"time elapsed FLD mean = {np.mean(time_FLD)}, std = {np.std(time_FLD)}")
print(f"time elapsed AUTH mean = {np.mean(time_AUTH)}, std = {np.std(time_AUTH)}")
print(f"time elapsed CT mean = {np.mean(time_CT)}, std = {np.std(time_CT)}")
print(f"time elapsed PQM mean = {np.mean(time_PQM)}, std = {np.std(time_PQM)}")