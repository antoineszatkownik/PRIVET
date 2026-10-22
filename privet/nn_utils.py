import torch


def gpu_nearest_neighbors(X, Y=None, k=1, distance="standard_euclidean", chunk_size=128,
                          device="cuda"):
    """
    k nearest neighbors of each row of X among the rows of Y (exact search).
    If Y is None, neighbors are searched within X, excluding each point itself.

    X, Y: tensors of shape (N_samples, N_features), i.e. samples flattened to 1D.
    distance: "hamming", "standard_euclidean", or "feature_normalized_euclidean"
    (euclidean divided by sqrt(N_features), cf. 4.1 https://arxiv.org/abs/2301.13188).

    Chunking avoids storing the full N_X x N_Y distance matrix.

    Returns (dists, indices), tensors of shape (N_X, k), sorted by distance.
    """
    device = torch.device(device)
    X = X.to(device)
    same = Y is None
    Y = X if same else Y.to(device)

    nX, dim = X.shape
    nY = Y.shape[0]

    if distance in ("standard_euclidean", "feature_normalized_euclidean"):
        x_sq = (X.float() ** 2).sum(dim=1)
        y_sq = x_sq if same else (Y.float() ** 2).sum(dim=1)
        factor = dim if distance == "feature_normalized_euclidean" else 1
    elif distance != "hamming":
        raise ValueError(f"unknown distance {distance!r}")

    # Hamming on 0/1 data as a matrix product, |x| + |y| - 2 x.y: exact in float32
    # (integers below 2^24) and much faster than comparing every feature pair.
    binary = (distance == "hamming" and dim < 2**24
              and bool(((X == 0) | (X == 1)).all()) and bool(((Y == 0) | (Y == 1)).all()))
    if binary:
        x_ones = X.float().sum(dim=1)
        y_ones = x_ones if same else Y.float().sum(dim=1)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False     # keep matrix products in full float32

    best_d = torch.full((nX, k), float("inf"), device=device)
    best_i = torch.full((nX, k), -1, device=device, dtype=torch.long)

    for i in range(0, nX, chunk_size):
        end_i = min(i + chunk_size, nX)
        Xc = X[i:end_i]
        chunk_d = torch.full((end_i - i, k), float("inf"), device=device)
        chunk_i = torch.full((end_i - i, k), -1, device=device, dtype=torch.long)

        for j in range(0, nY, chunk_size):
            end_j = min(j + chunk_size, nY)
            Yc = Y[j:end_j]

            if binary:
                d = x_ones[i:end_i, None] + y_ones[None, j:end_j] - 2 * torch.mm(Xc.float(), Yc.float().t())
            elif distance == "hamming":
                # exact integer mismatch counts
                d = (Xc.unsqueeze(1) != Yc.unsqueeze(0)).to(torch.int32).sum(dim=2).to(torch.float32)
            else:
                xy = torch.mm(Xc.float(), Yc.float().t())
                sq = (x_sq[i:end_i, None] + y_sq[None, j:end_j] - 2 * xy) / factor
                d = torch.sqrt(torch.clamp(sq, min=0.0))

            if same:
                # mask self-distances
                rows = torch.arange(i, end_i, device=device)
                cols = torch.arange(j, end_j, device=device)
                d.masked_fill_(rows.unsqueeze(1) == cols.unsqueeze(0), float("inf"))

            # merge into top-k
            concat_d = torch.cat([chunk_d, d], dim=1)
            idx_block = torch.arange(j, end_j, device=device).expand(end_i - i, -1)
            concat_i = torch.cat([chunk_i, idx_block], dim=1)
            chunk_d, pos = torch.topk(concat_d, k, largest=False, sorted=True)
            chunk_i = torch.gather(concat_i, 1, pos)

        best_d[i:end_i] = chunk_d
        best_i[i:end_i] = chunk_i

    torch.backends.cuda.matmul.allow_tf32 = tf32

    return best_d.cpu(), best_i.cpu()
