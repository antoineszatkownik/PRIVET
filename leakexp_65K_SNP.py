#PRIVET ONLY

from p_utils_v6 import *
import time
import torch

ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

###############
###LOAD DATA###
###############
ROOT_PATH = "your/path"
dat=np.load(f"{ROOT_PATH}/65k_all_labels.npy",allow_pickle=True)

np.random.seed(42)
np.random.shuffle(dat)
dat = dat[:3*(dat.shape[0]//3)] #5006 is not divisible by 3, 5004 is
train, test, synth = dat[:dat.shape[0]//3,3:], dat[dat.shape[0]//3:2*(dat.shape[0]//3),3:], dat[2*(dat.shape[0]//3):,3:]
train, test, synth = train.astype(int), test.astype(int), synth.astype(int)

train_torch = torch.tensor(train, dtype = torch.uint8)
test_torch = torch.tensor(test, dtype = torch.uint8)
synth_torch = torch.tensor(synth, dtype = torch.uint8)

N = train.shape[0]

###################################
###INITIALIZE LEAKAGE PARAMETERS###
###################################
fake_range = np.linspace(0,0.4,31)
fake_range[0] = 0.001

copy_range = np.linspace(0,0.2,31)
copy_range[0] = 0.001

#############################
###INITIALIZE PRIVACY MAPS###
#############################
heatmap_delta_pi=np.zeros((len(fake_range),len(copy_range)))
heatmap_NPL=np.zeros((len(fake_range),len(copy_range)))

tp_grid_npl = np.zeros((len(fake_range), len(copy_range)))
fp_grid_npl = np.zeros((len(fake_range), len(copy_range)))
tn_grid_npl = np.zeros((len(fake_range), len(copy_range)))
fn_grid_npl = np.zeros((len(fake_range), len(copy_range)))

############################
## COMPUTE 1-NN distances ##
############################
#Train-Train
dist_NN_tr_tr = gpu_nearest_neighbors(train_torch, k=1,distance='hamming',chunk_size=128,device=device.type,verbose=False)
p_tr_tr_NN_dist, p_tr_tr_NN_idx = sorting(dist_NN_tr_tr)

#Synth-Train
dist_NN_syn_tr_INIT = gpu_nearest_neighbors(synth_torch, train_torch, k=1,distance='hamming',chunk_size=128,device=device.type,verbose=False)
p_syn_tr_NN_dist_INIT, p_syn_tr_NN_idx_INIT = sorting(dist_NN_syn_tr_INIT)
indices_s_tr_INIT = dist_NN_syn_tr_INIT[1].numpy().squeeze(1)

############################
## FIT CDF on Train-Train ##
############################

partition_start = 0.01
partition_end = 0.2

start = int(np.ceil(partition_start*N)) #int(0.01*N) # if N is small start = 0 --> problem with log
end = int(partition_end*N)

# Fit parameters (adjust start/end indices to avoid extremes)
intercept, alpha, std_err_intercept, std_err_alpha, sigma_Y_pred = fit_nearest_neighbor_cdf(p_tr_tr_NN_dist.numpy().reshape(-1,), start_idx=start, end_idx=end)

print(f"Estimated intercept = {intercept:.2f} ± {std_err_intercept:.2f}")
print(f"Estimated alpha = {alpha:.2f} ± {std_err_alpha:.2f}")

#######################
###FILL PRIVACY MAPS###
#######################
start = time.time()
time_lst = []
for i,f_fake in enumerate(fake_range):
    print(f_fake)
    n = int(np.ceil(N*f_fake))
    flist=torch.zeros((N,),dtype=bool)
    flist[:n]=True
    for j,f_copy in enumerate(copy_range):

        fake = generate_fake_synth(train, synth, indices_s_tr_INIT, f_fake=f_fake, f_copy=f_copy)

        start_i_j = time.time()
        store_in_mat, p_syn_tr_NN_dist, p_syn_te_NN_dist = PRIVET(train_torch, test_torch, fake, intercept, alpha, renormalization = None, distance='hamming', device=device.type, groundtruth = flist)
        end_i_j = time.time()
        time_lst.append(end_i_j-start_i_j)

        delta_pi = store_in_mat[:,0]
        flist_bis = store_in_mat[:,-1].astype(bool)

        pred_with_delta_pi = delta_pi<=threshold

        tp_deltapi, fn_deltapi, fp_deltapi, tn_deltapi = get_predictions(pred_with_delta_pi, flist_bis)

        heatmap_delta_pi[i,j] = np.mean(delta_pi)
        heatmap_NPL[i,j] = pred_with_delta_pi.sum()

        tp_grid_npl[i, j] = tp_deltapi
        fp_grid_npl[i, j] = fp_deltapi
        tn_grid_npl[i, j] = tn_deltapi
        fn_grid_npl[i, j] = fn_deltapi
        
        if j%10==0:
            print(rf"Δπ f_fake={f_fake}, f_copy={f_copy}, tp={tp_deltapi}, fp={fp_deltapi}, tn={tn_deltapi}, fn={fn_deltapi}")

        del fake

end = time.time()
print(f"{end-start}s")

time_lst = np.array(time_lst)
print(f"mean time={np.mean(time_lst)}, std time={np.std(time_lst)}")

#9186.533095121384s
#mean time=8.852286925300971, std time=0.007254868125433914

np.save('heatmap_delta_pi_leakexp_65K_SNP.npy',heatmap_delta_pi)
np.save('heatmap_NPL_leakexp_65K_SNP.npy',heatmap_NPL)

np.save('heatmap_NPL_tp_leakexp_65K_SNP.npy',tp_grid_npl)
np.save('heatmap_NPL_tn_leakexp_65K_SNP.npy',tn_grid_npl)
np.save('heatmap_NPL_fp_leakexp_65K_SNP.npy',fp_grid_npl)
np.save('heatmap_NPL_fn_leakexp_65K_SNP.npy',fn_grid_npl)