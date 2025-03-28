# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2007-2020 The scikit-learn developers.

# BSD 3-Clause License

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# NME-SC clustering is based on the implementation from the paper
# https://arxiv.org/pdf/2003.02405.pdf and the implementation from
# https://github.com/tango4j/Auto-Tuning-Spectral-Clustering.

from collections import Counter
from typing import Dict, List

import torch
from torch.linalg import eigh


@torch.jit.script
def cos_similarity(a: torch.Tensor, b: torch.Tensor, eps=torch.tensor(3.5e-4)):
    """
    Args:
        a: (torch.tensor)
            Matrix containing speaker representation vectors. (N x embedding_dim)
        b: (torch.tensor)
            Matrix containing speaker representation vectors. (N x embedding_dim)
    Returns:
        res (torch.tensor)
            N by N matrix containing the cosine similarities of the values.
    """
    a_norm = a / (torch.norm(a, dim=1).unsqueeze(1) + eps)
    b_norm = b / (torch.norm(a, dim=1).unsqueeze(1) + eps)
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    res.fill_diagonal_(1)
    return res


@torch.jit.script
def ScalerMinMax(X: torch.Tensor):
    """
    Min-max scale the input affinity matrix X, which will lead to a dynamic range of
    [0, 1].

    Args:
        X: (torch.tensor)
            Matrix containing cosine similarity values among embedding vectors (N x N)

    Returns:
        v_norm: (torch.tensor)
            Min-max normalized value of X.
    """
    v_min, v_max = X.min(), X.max()
    v_norm = (X - v_min) / (v_max - v_min)
    return v_norm


@torch.jit.script
def getEuclideanDistance(specEmbA: torch.Tensor, specEmbB: torch.Tensor, device: torch.device = torch.device('cpu')):
    """
    Args:
        specEmbA: (torch.tensor)
            Matrix containing spectral embedding vectors from eigenvalue decomposition (N x embedding_dim).

        specEmbB: (torch.tensor)
            Matrix containing spectral embedding vectors from eigenvalue decomposition (N x embedding_dim).

    Returns:
        dis: (torch.tensor)
            Euclidean distance values of the two sets of spectral embedding vectors.
    """
    specEmbA, specEmbB = specEmbA.to(device), specEmbB.to(device)
    A, B = specEmbA.unsqueeze(dim=1), specEmbB.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis


@torch.jit.script
def kmeans_plusplus_torch(
    X: torch.Tensor,
    n_clusters: int,
    random_state: int,
    n_local_trials: int = 30,
    device: torch.device = torch.device('cpu'),
):
    """
    Choose initial centroids for initializing k-means algorithm. The performance of
    k-means algorithm can vary significantly by the initial centroids. To alleviate
    this problem, k-means++ algorithm chooses initial centroids based on the probability
    proportional to the distance from the formally chosen centroids. The centroids
    selected by k-means++ algorithm improve the chance of getting more accurate and
    stable clustering results. The overall implementation of k-means++ algorithm is
    inspired by the numpy based k-means++ implementation in:
        https://github.com/scikit-learn/scikit-learn

    Originally, the implementation of the k-means++ algorithm in scikit-learn is based
    on the following research article:
        Arthur, David, and Sergei Vassilvitskii. k-means++: The advantages of careful
        seeding. Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
        algorithms, Society for Industrial and Applied Mathematics (2007)

    Args:
        X: (torch.tensor)
            Matrix containing cosine similarity values among embedding vectors (N x N)

        n_clusters: (int)
            Maximum number of speakers for estimating number of speakers.
            Shows stable performance under 20.

        random_state: (int)
            Seed variable for setting up a random state.

        n_local_trials: (int)
            Number of trials for creating initial values of the center points.

        device: (torch.device)
            Torch device variable.

    Returns:
        centers: (torch.tensor)
            The coordinates for center points that are used for initializing k-means algorithm.

        indices: (torch.tensor)
            The indices of the best candidate center points.
    """
    torch.manual_seed(random_state)
    X = X.to(device)
    n_samples, n_features = X.shape

    centers = torch.zeros(n_clusters, n_features, dtype=X.dtype)
    center_id = torch.randint(0, n_samples, (1,)).long()
    indices = torch.full([n_clusters,], -1, dtype=torch.int)

    centers[0] = X[center_id].squeeze(0)
    indices[0] = center_id.squeeze(0)

    centers = centers.to(device)
    closest_dist_diff = centers[0, None].repeat(1, X.shape[0]).view(X.shape[0], -1) - X
    closest_dist_sq = closest_dist_diff.pow(2).sum(dim=1).unsqueeze(dim=0)
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = torch.rand(n_local_trials) * current_pot.item()

        if len(closest_dist_sq.shape) > 1:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=1)[0]
        else:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=0)

        candidate_ids = torch.searchsorted(torch_cumsum, rand_vals.to(device))

        N_ci = candidate_ids.shape[0]
        distance_diff = X[candidate_ids].repeat(1, X.shape[0]).view(X.shape[0] * N_ci, -1) - X.repeat(N_ci, 1)
        distance = distance_diff.pow(2).sum(dim=1).view(N_ci, -1)
        distance_to_candidates = torch.minimum(closest_dist_sq, distance)
        candidates_pot = distance_to_candidates.sum(dim=1)

        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    return centers, indices


@torch.jit.script
def kmeans_torch(
    X: torch.Tensor,
    num_clusters: int,
    threshold: float = 1e-4,
    iter_limit: int = 15,
    random_state: int = 0,
    device: torch.device = torch.device('cpu'),
):
    """
    Run k-means algorithm on the given set of spectral embeddings in X. The threshold
    and iter_limit variables are set to show the best performance on speaker diarization
    tasks. The overall implementation of k-means algorithm is inspired by the k-means
    algorithm implemented in https://github.com/scikit-learn/scikit-learn.

    References:
        Arthur, David, and Sergei Vassilvitskii. k-means++: The advantages of careful
        seeding. Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
        algorithms, Society for Industrial and Applied Mathematics (2007).

    Args:
        X: (torch.tensor)
            Cosine similarity matrix calculated from speaker embeddings

        num_clusters: (int)
            The estimated number of speakers.

        threshold: (float)
            This threshold limits the change of center values. If the square of
            the center shift values are bigger than this threshold, the iteration stops.

        iter_limit: (int)
            The maximum number of iterations that is allowed by the k-means algorithm.

        device: (torch.device)
            Torch device variable

    Returns:
        selected_cluster_indices: (torch.tensor)
            The assigned cluster labels from the k-means clustering.
    """
    # Convert tensor type to float
    X = X.float().to(device)
    input_size = X.shape[0]

    # Initialize the cluster centers with kmeans_plusplus algorithm.
    plusplus_init_states = kmeans_plusplus_torch(X, n_clusters=num_clusters, random_state=random_state, device=device)
    centers = plusplus_init_states[0]

    iter_count = 0
    selected_cluster_indices = torch.zeros(input_size).int()

    for iter_count in range(iter_limit):
        euc_dist = getEuclideanDistance(X, centers, device=device)

        if len(euc_dist.shape) <= 1:
            break
        else:
            selected_cluster_indices = torch.argmin(euc_dist, dim=1)

        center_inits = centers.clone()

        for index in range(num_clusters):
            selected_cluster = torch.nonzero(selected_cluster_indices == index).squeeze().to(device)
            chosen_indices = torch.index_select(X, 0, selected_cluster)

            if chosen_indices.shape[0] == 0:
                chosen_indices = X[torch.randint(len(X), (1,))]

            centers[index] = chosen_indices.mean(dim=0)

        # Calculate the delta from center_inits to centers
        center_delta_pow = torch.pow((centers - center_inits), 2)
        center_shift_pow = torch.pow(torch.sum(torch.sqrt(torch.sum(center_delta_pow, dim=1))), 2)

        # If the cluster centers are not changing significantly, stop the loop.
        if center_shift_pow < threshold:
            break

    return selected_cluster_indices


@torch.jit.script
def getTheLargestComponent(affinity_mat: torch.Tensor, seg_index: int, device: torch.device):
    """
    Find the largest affinity_mat connected components for each given node.
    This is for checking whether the affinity_mat is fully connected.

    Args:
        affinity_mat: (torch.tensor)
            A square matrix (tensor) containing normalized cosine distance values

        seg_index: (int)
            The segment index that is targeted to be explored.
    Returns:
        connected_nodes: (torch.tensor)
            A tensor containing booleans that indicate whether the node is connected.

    """
    num_of_segments = affinity_mat.shape[0]

    connected_nodes = torch.zeros(num_of_segments, dtype=torch.bool).to(device)
    nodes_to_explore = torch.zeros(num_of_segments, dtype=torch.bool).to(device)

    nodes_to_explore[seg_index] = True
    for k in range(num_of_segments):
        last_num_component = connected_nodes.sum()
        torch.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break

        indices = (nodes_to_explore == torch.tensor(True)).nonzero().t().squeeze()
        if len(indices.size()) == 0:
            indices = indices.unsqueeze(0)
        for i in indices:
            neighbors = affinity_mat[i]
            torch.logical_or(nodes_to_explore, neighbors.squeeze(0), out=nodes_to_explore)
    return connected_nodes


@torch.jit.script
def isGraphFullyConnected(affinity_mat: torch.Tensor, device: torch.device):
    """
    Check whether the given affinity matrix is a fully connected graph.
    """
    return getTheLargestComponent(affinity_mat, 0, device).sum() == affinity_mat.shape[0]


@torch.jit.script
def getKneighborsConnections(affinity_mat: torch.Tensor, p_value: int):
    """
    Binarize top-p values for each row from the given affinity matrix.
    """
    binarized_affinity_mat = torch.zeros_like(affinity_mat)
    for i in range(affinity_mat.shape[0]):
        line = affinity_mat[i, :]
        sorted_idx = torch.argsort(line, descending=True)
        indices = sorted_idx[:p_value]
        binarized_affinity_mat[indices, i] = 1

    return binarized_affinity_mat


@torch.jit.script
def getAffinityGraphMat(affinity_mat_raw: torch.Tensor, p_value: int):
    """
    Calculate a binarized graph matrix and
    symmetrize the binarized graph matrix.
    """
    X = getKneighborsConnections(affinity_mat_raw, p_value)
    symm_affinity_mat = 0.5 * (X + X.T)
    return symm_affinity_mat


@torch.jit.script
def getMinimumConnection(mat: torch.Tensor, max_N: torch.Tensor, n_list: torch.Tensor, device: torch.device):
    """
    Generate connections until fully connect all the nodes in the graph.
    If the graph is not fully connected, it might generate inaccurate results.
    """
    p_value = torch.tensor(1)
    affinity_mat = getAffinityGraphMat(mat, p_value)
    for i, p_value in enumerate(n_list):
        fully_connected = isGraphFullyConnected(affinity_mat, device)
        affinity_mat = getAffinityGraphMat(mat, p_value)
        if fully_connected or p_value > max_N:
            break

    return affinity_mat, p_value


@torch.jit.script
def getRepeatedList(mapping_argmat: torch.Tensor, score_mat_size: torch.Tensor):
    """
    Count the numbers in the mapping dictionary and create lists that contain
    repeated indices that will be used for creating a repeated affinity matrix.
    This repeated matrix is then used for fusing multiple affinity values.
    """
    repeat_list = torch.zeros(score_mat_size, dtype=torch.int32)
    idxs, counts = torch.unique(mapping_argmat, return_counts=True)
    repeat_list[idxs] = counts.int()
    return repeat_list


def get_argmin_mat(uniq_scale_dict: dict):
    """
    Calculate the mapping between the base scale and other scales. A segment from a longer scale is
    repeatedly mapped to a segment from a shorter scale or the base scale.

    Args:
        uniq_scale_dict (dict) :
            Dictionary of embeddings and timestamps for each scale.

    Returns:
        session_scale_mapping_dict (dict) :
            Dictionary containing argmin arrays indexed by scale index.
    """
    scale_list = sorted(list(uniq_scale_dict.keys()))
    segment_anchor_dict = {}
    for scale_idx in scale_list:
        time_stamp_list = uniq_scale_dict[scale_idx]['time_stamps']
        time_stamps_float = torch.tensor([[float(x.split()[0]), float(x.split()[1])] for x in time_stamp_list])
        segment_anchor_dict[scale_idx] = torch.mean(time_stamps_float, dim=1)

    base_scale_idx = max(scale_list)
    base_scale_anchor = segment_anchor_dict[base_scale_idx]
    session_scale_mapping_dict = {}
    for scale_idx in scale_list:
        curr_scale_anchor = segment_anchor_dict[scale_idx]
        curr_mat = torch.tile(curr_scale_anchor, (base_scale_anchor.shape[0], 1))
        base_mat = torch.tile(base_scale_anchor, (curr_scale_anchor.shape[0], 1)).t()
        argmin_mat = torch.argmin(torch.abs(curr_mat - base_mat), dim=1)
        session_scale_mapping_dict[scale_idx] = argmin_mat
    return session_scale_mapping_dict


def getMultiScaleCosAffinityMatrix(uniq_embs_and_timestamps: dict, device: torch.device = torch.device('cpu')):
    """
    Calculate cosine similarity values among speaker embeddings for each scale then
    apply multiscale weights to calculate the fused similarity matrix.

    Args:
        uniq_embs_and_timestamps: (dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_embs_and_timestamps contains only one scale, single scale diarization
            is performed.

    Returns:
        fused_sim_d (torch.tensor):
            This function generates an affinity matrix that is obtained by calculating
            the weighted sum of the affinity matrices from the different scales.

        base_scale_emb (torch.tensor):
            The base scale embedding (the embeddings from the finest scale)
    """
    uniq_scale_dict = uniq_embs_and_timestamps['scale_dict']
    base_scale_idx = max(uniq_scale_dict.keys())
    base_scale_emb = uniq_scale_dict[base_scale_idx]['embeddings']
    multiscale_weights = uniq_embs_and_timestamps['multiscale_weights'].float().to(device)
    score_mat_list, repeated_tensor_list = [], []
    session_scale_mapping_dict = get_argmin_mat(uniq_scale_dict)
    for scale_idx in sorted(uniq_scale_dict.keys()):
        mapping_argmat = session_scale_mapping_dict[scale_idx]
        emb_t = uniq_scale_dict[scale_idx]['embeddings'].half().to(device)
        score_mat_torch = getCosAffinityMatrix(emb_t)
        repeat_list = getRepeatedList(mapping_argmat, torch.tensor(score_mat_torch.shape[0])).to(device)
        repeated_tensor_0 = torch.repeat_interleave(score_mat_torch, repeats=repeat_list, dim=0)
        repeated_tensor_1 = torch.repeat_interleave(repeated_tensor_0, repeats=repeat_list, dim=1)
        repeated_tensor_list.append(repeated_tensor_1)
    repp = torch.stack(repeated_tensor_list).float()
    fused_sim_d = torch.matmul(repp.permute(2, 1, 0), multiscale_weights.t()).squeeze(2).t()
    return fused_sim_d, base_scale_emb


@torch.jit.script
def getCosAffinityMatrix(_emb: torch.Tensor):
    """
    Calculate cosine similarity values among speaker embeddings then min-max normalize
    the affinity matrix.
    """
    emb = _emb.half()
    sim_d = cos_similarity(emb, emb)
    sim_d = ScalerMinMax(sim_d)
    return sim_d


@torch.jit.script
def getLaplacian(X: torch.Tensor):
    """
    Calculate a laplacian matrix from an affinity matrix X.
    """
    X.fill_diagonal_(0)
    D = torch.sum(torch.abs(X), dim=1)
    D = torch.diag_embed(D)
    L = D - X
    return L


@torch.jit.script
def eigDecompose(laplacian: torch.Tensor, cuda: bool, device: torch.device = torch.device('cpu')):
    """
    Calculate eigenvalues and eigenvectors from the Laplacian matrix.
    """
    if cuda:
        if device is None:
            device = torch.cuda.current_device()
        laplacian = laplacian.float().to(device)
    else:
        laplacian = laplacian.float()
    lambdas, diffusion_map = eigh(laplacian)
    return lambdas, diffusion_map


@torch.jit.script
def getLamdaGaplist(lambdas: torch.Tensor):
    """
    Calculate the gaps between lambda values.
    """
    if torch.is_complex(lambdas):
        lambdas = torch.real(lambdas)
    return lambdas[1:] - lambdas[:-1]


def addAnchorEmb(emb: torch.Tensor, anchor_sample_n: int, anchor_spk_n: int, sigma: float):
    """
    Add randomly generated synthetic embeddings to make eigen analysis more stable.
    We refer to these embeddings as anchor embeddings.

    emb (torch.tensor):
        The input embedding from the embedding extractor.

    anchor_sample_n (int):
        Number of embedding samples per speaker.
        anchor_sample_n = 10 is recommended.

    anchor_spk_n (int):
        Number of speakers for synthetic embedding.
        anchor_spk_n = 3 is recommended.

    sigma (int):
        The amplitude of synthetic noise for each embedding vector.
        If the sigma value is too small, under-counting could happen.
        If the sigma value is too large, over-counting could happen.
        sigma = 50 is recommended.

    """
    emb_dim = emb.shape[1]
    std_org = torch.std(emb, dim=0)
    new_emb_list = []
    for _ in range(anchor_spk_n):
        emb_m = torch.tile(torch.randn(1, emb_dim), (anchor_sample_n, 1))
        emb_noise = torch.randn(anchor_sample_n, emb_dim).T
        emb_noise = torch.matmul(
            torch.diag(std_org), emb_noise / torch.max(torch.abs(emb_noise), dim=0)[0].unsqueeze(0)
        ).T
        emb_gen = emb_m + sigma * emb_noise
        new_emb_list.append(emb_gen)

    new_emb_list.append(emb)
    new_emb_np = torch.vstack(new_emb_list)
    return new_emb_np


def getEnhancedSpeakerCount(
    emb: torch.Tensor,
    cuda: bool,
    random_test_count: int = 5,
    anchor_spk_n: int = 3,
    anchor_sample_n: int = 10,
    sigma: float = 50,
):
    """
    Calculate the number of speakers using NME analysis with anchor embeddings.

    emb (torch.Tensor):
        The input embedding from the embedding extractor.

    cuda (bool):
        Use cuda for the operations if cuda==True.

    random_test_count (int):
        Number of trials of the enhanced counting with randomness.
        The higher the count, the more accurate the enhanced counting is.

    anchor_spk_n (int):
        Number of speakers for synthetic embedding.
        anchor_spk_n = 3 is recommended.

    anchor_sample_n (int):
        Number of embedding samples per speaker.
        anchor_sample_n = 10 is recommended.

    sigma (float):
        The amplitude of synthetic noise for each embedding vector.
        If the sigma value is too small, under-counting could happen.
        If the sigma value is too large, over-counting could happen.
        sigma = 50 is recommended.

    """
    est_num_of_spk_list = []
    for seed in range(random_test_count):
        torch.manual_seed(seed)
        emb_aug = addAnchorEmb(emb, anchor_sample_n, anchor_spk_n, sigma)
        mat = getCosAffinityMatrix(emb_aug)
        nmesc = NMESC(
            mat,
            max_num_speaker=emb.shape[0],
            max_rp_threshold=0.15,
            sparse_search=True,
            sparse_search_volume=50,
            fixed_thres=-1.0,
            NME_mat_size=300,
            cuda=cuda,
        )
        est_num_of_spk, _ = nmesc.NMEanalysis()
        est_num_of_spk_list.append(est_num_of_spk)
    ctt = Counter(est_num_of_spk_list)
    comp_est_num_of_spk = max(ctt.most_common(1)[0][0] - anchor_spk_n, 1)
    return comp_est_num_of_spk


@torch.jit.script
def estimateNumofSpeakers(affinity_mat: torch.Tensor, max_num_speaker: int, cuda: bool = False):
    """
    Estimate the number of speakers using eigendecomposition on the Laplacian Matrix.

    Args:
        affinity_mat: (torch.tensor)
            N by N affinity matrix

        max_num_speaker: (int)
            Maximum number of clusters to consider for each session

        cuda: (bool)
            If cuda available eigendecomposition is computed on GPUs.

    Returns:
        num_of_spk: (torch.tensor)
            The estimated number of speakers

        lambdas: (torch.tensor)
            The lambda values from eigendecomposition

        lambda_gap: (torch.tensor)
            The gap between the lambda values from eigendecomposition
    """
    laplacian = getLaplacian(affinity_mat)
    lambdas, _ = eigDecompose(laplacian, cuda)
    lambdas = torch.sort(lambdas)[0]
    lambda_gap = getLamdaGaplist(lambdas)
    num_of_spk = torch.argmax(lambda_gap[: min(max_num_speaker, lambda_gap.shape[0])]) + 1
    return num_of_spk, lambdas, lambda_gap


@torch.jit.script
class SpectralClustering:
    """
    Perform spectral clustering by calculating spectral embeddings then run k-means clustering
    algorithm on the spectral embeddings.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        random_state: int = 0,
        random_trial: int = 1,
        cuda: bool = False,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize the variables needed for spectral clustering and k-means++.

        Args:
            n_clusters (int):
                Number of the estimated (or oracle) number of speakers

            random_state (int):
                Random seed that determines a random state of k-means initialization.

            random_trial (int):
                Number of trials with different random seeds for k-means initialization.
                k-means++ algorithm is executed for multiple times then the final result
                is obtained by taking a majority vote.

            cuda (bool):
                if cuda=True, spectral clustering is done on GPU.

            device (torch.device):
                Torch device variable

        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.random_trial = max(random_trial, 1)
        self.cuda = cuda
        self.device = device

    def predict(self, X):
        """
        Call self.clusterSpectralEmbeddings() function to predict cluster labels.

        Args:
            X (torch.tensor):
                Affinity matrix input

        Returns:
            labels (torch.tensor):
                clustering label output
        """
        if X.shape[0] != X.shape[1]:
            raise ValueError("The affinity matrix is not a square matrix.")
        labels = self.clusterSpectralEmbeddings(X, cuda=self.cuda, device=self.device)
        return labels

    def clusterSpectralEmbeddings(self, affinity, cuda: bool = False, device: torch.device = torch.device('cpu')):
        """
        Perform k-means clustering on spectral embeddings. To alleviate the effect of randomness,
        k-means clustering is performed for (self.random_trial) times then the final labels are obtained
        by taking a majority vote. If speed is the major concern, self.random_trial should be set to 1.
        random_trial=30 is recommended to see an improved result.

        Args:
            affinity (torch.tensor):
                Affinity matrix input
            cuda (torch.bool):
                Use cuda for spectral clustering if cuda=True
            device (torch.device):
                Torch device variable

        Returns:
            labels (torch.tensor):
                clustering label output

        """
        spectral_emb = self.getSpectralEmbeddings(affinity, n_spks=self.n_clusters, cuda=cuda)
        labels_set = []
        for random_state_seed in range(self.random_state, self.random_state + self.random_trial):
            _labels = kmeans_torch(
                X=spectral_emb, num_clusters=self.n_clusters, random_state=random_state_seed, device=device
            )
            labels_set.append(_labels)
        stacked_labels = torch.stack(labels_set)
        label_index = torch.mode(torch.mode(stacked_labels, 0)[1])[0]
        labels = stacked_labels[label_index]
        return labels

    def getSpectralEmbeddings(self, affinity_mat: torch.Tensor, n_spks: int = 8, cuda: bool = False):
        """
        Calculate eigenvalues and eigenvectors to extract spectral embeddings.

        Args:
            affinity (torch.tensor):
                Affinity matrix input
            cuda (torch.bool):
                Use cuda for spectral clustering if cuda=True
            device (torch.device):
                Torch device variable

        Returns:
            labels (torch.Tensor):
                clustering label output
        """
        laplacian = getLaplacian(affinity_mat)
        lambdas_, diffusion_map_ = eigDecompose(laplacian, cuda)
        diffusion_map = diffusion_map_[:, :n_spks]
        inv_idx = torch.arange(diffusion_map.size(1) - 1, -1, -1).long()
        embedding = diffusion_map.T[inv_idx, :]
        return embedding[:n_spks].T


@torch.jit.script
class NMESC:
    """
    Normalized Maximum Eigengap based Spectral Clustering (NME-SC)
    uses Eigengap analysis to get an estimated p-value for
    affinity binarization and an estimated number of speakers.

    p_value (also referred to as p_neighbors) is for taking
    top p number of affinity values and convert those to 1 while
    convert the rest of values to 0.

    p_value can be also tuned on a development set without performing
    NME-analysis. Fixing p_value brings about significantly faster clustering
    speed, but the performance is limited to the development set.

    References:
        Tae Jin Park et al., Auto-Tuning Spectral Clustering for Speaker Diarization
        Using Normalized Maximum Eigengap, IEEE Signal Processing Letters 27 (2019),
        https://arxiv.org/abs/2003.02405

    Args:
        Please refer to def __init__().

    Methods:
        NMEanalysis():
            Performs NME-analysis to estimate p_value and the number of speakers

        subsampleAffinityMat(NME_mat_size):
            Subsamples the number of speakers to reduce the computational load

        getPvalueList():
            Generates a list containing p-values that need to be examined.

        getEigRatio(p_neighbors):
            Calculates g_p, which is a ratio between p_neighbors and the maximum eigengap

        getLamdaGaplist(lambdas):
            Calculates lambda gap values from an array contains lambda values

        estimateNumofSpeakers(affinity_mat):
            Estimates the number of speakers using lambda gap list

    """

    def __init__(
        self,
        mat,
        max_num_speaker: int = 10,
        max_rp_threshold: float = 0.15,
        sparse_search: bool = True,
        sparse_search_volume: int = 30,
        use_subsampling_for_NME: bool = True,
        fixed_thres: float = 0.0,
        cuda: bool = False,
        NME_mat_size: int = 512,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            mat: (torch.tensor)
                Cosine similarity matrix calculated from the provided speaker embeddings.

            max_num_speaker: (int)
                Maximum number of speakers for estimating number of speakers.
                Shows stable performance under 20.

            max_rp_threshold: (float)
                Limits the range of parameter search.
                Clustering performance can vary depending on this range.
                Default is 0.25.

            sparse_search: (bool)
                To increase the speed of parameter estimation, sparse_search=True
                limits the number of p_values we search.

            sparse_search_volume: (int)
                Number of p_values we search during NME analysis.
                Default is 30. The lower the value, the faster NME-analysis becomes.
                However, a value lower than 20 might cause a poor parameter estimation.

            use_subsampling_for_NME: (bool)
                Use subsampling to reduce the calculational complexity.
                Default is True.

            fixed_thres: (float or None)
                A fixed threshold which can be used instead of estimating the
                threshold with NME analysis. If fixed_thres is float,
                it skips the NME analysis part.

            cuda (bool)
                Use cuda for Eigen decomposition if cuda=True.

            NME_mat_size: (int)
                Targeted size of matrix for NME analysis.


        """
        self.max_num_speaker: int = max_num_speaker
        self.max_rp_threshold = max_rp_threshold
        self.use_subsampling_for_NME = use_subsampling_for_NME
        self.NME_mat_size: int = NME_mat_size
        self.sparse_search = sparse_search
        self.sparse_search_volume = sparse_search_volume
        self.fixed_thres: float = fixed_thres
        self.cuda: bool = cuda
        self.eps = 1e-10
        self.max_N = torch.tensor(0)
        self.mat = mat
        self.p_value_list: torch.Tensor = torch.tensor(0)
        self.device = device

    def NMEanalysis(self):
        """
        Subsample the input matrix to reduce the computational load.
        """
        if self.use_subsampling_for_NME:
            subsample_ratio = self.subsampleAffinityMat(self.NME_mat_size)
        else:
            subsample_ratio = torch.tensor(1)

        # Scans p_values and find a p_value that generates
        # the smallest g_p value.
        eig_ratio_list = []
        est_spk_n_dict: Dict[int, torch.Tensor] = {}
        self.p_value_list = self.getPvalueList()
        for p_value in self.p_value_list:
            est_num_of_spk, g_p = self.getEigRatio(p_value)
            est_spk_n_dict[p_value.item()] = est_num_of_spk
            eig_ratio_list.append(g_p)
        index_nn = torch.argmin(torch.tensor(eig_ratio_list))
        rp_p_value = self.p_value_list[index_nn]
        affinity_mat = getAffinityGraphMat(self.mat, rp_p_value)

        # Checks whether the affinity graph is fully connected.
        # If not, it adds a minimum number of connections to make it fully connected.
        if not isGraphFullyConnected(affinity_mat, device=self.device):
            affinity_mat, rp_p_value = getMinimumConnection(
                self.mat, self.max_N, self.p_value_list, device=self.device
            )

        p_hat_value = (subsample_ratio * rp_p_value).type(torch.int)
        est_num_of_spk = est_spk_n_dict[rp_p_value.item()]
        return est_num_of_spk, p_hat_value

    def subsampleAffinityMat(self, NME_mat_size: int):
        """
        Perform subsampling of affinity matrix.
        This subsampling is for calculational complexity, not for performance.
        The smaller NME_mat_size is,
            - the bigger the chance of missing a speaker.
            - the faster p-value estimation speed (based on eigen decomposition).

        The recommended NME_mat_size is 250~750.
        However, if there are speakers who speak for very short period of time in the recording,
        this subsampling might make the system miss underrepresented speakers.
        Use this variable with caution.

        Args:
            NME_mat_size: (int)
                The targeted matrix size

        Returns:
            subsample_ratio : (float)
                The ratio between NME_mat_size and the original matrix size

        """
        subsample_ratio = torch.max(torch.tensor(1), torch.tensor(self.mat.shape[0] / NME_mat_size)).type(torch.int)
        self.mat = self.mat[:: subsample_ratio.item(), :: subsample_ratio.item()]
        return subsample_ratio

    def getEigRatio(self, p_neighbors: int):
        """
        For a given p_neighbors value, calculate g_p, which is a ratio between p_neighbors and the 
        maximum eigengap values.
        References:
            Tae Jin Park et al., Auto-Tuning Spectral Clustering for Speaker Diarization Using 
            Normalized Maximum Eigengap, IEEE Signal Processing Letters 27 (2019),
            https://arxiv.org/abs/2003.02405

        Args:
            p_neighbors: (int)
                Determines how many binary graph connections we want to keep for each row.

        Returns:
            est_num_of_spk: (int)
                Estimated number of speakers

            g_p: (float)
                The ratio between p_neighbors value and the maximum eigen gap value.
        """
        affinity_mat = getAffinityGraphMat(self.mat, p_neighbors)
        est_num_of_spk, lambdas, lambda_gap_list = estimateNumofSpeakers(affinity_mat, self.max_num_speaker, self.cuda)
        arg_sorted_idx = torch.argsort(lambda_gap_list[: self.max_num_speaker], descending=True)
        max_key = arg_sorted_idx[0]
        max_eig_gap = lambda_gap_list[max_key] / (max(lambdas) + self.eps)
        g_p = (p_neighbors / self.mat.shape[0]) / (max_eig_gap + self.eps)
        return est_num_of_spk, g_p

    def getPvalueList(self):
        """
        Generates a p-value (p_neighbour) list for searching.
        """
        if self.fixed_thres > 0.0:
            p_value_list = torch.floor(torch.tensor(self.mat.shape[0] * self.fixed_thres)).type(torch.int)
            self.max_N = p_value_list[0]
        else:
            self.max_N = torch.floor(torch.tensor(self.mat.shape[0] * self.max_rp_threshold)).type(torch.int)
            if self.sparse_search:
                N = torch.min(self.max_N, torch.tensor(self.sparse_search_volume).type(torch.int))
                p_value_list = torch.unique(torch.linspace(start=1, end=self.max_N, steps=N).type(torch.int))
            else:
                p_value_list = torch.arange(1, self.max_N)

        return p_value_list


def COSclustering(
    uniq_embs_and_timestamps,
    oracle_num_speakers=None,
    max_num_speaker: int = 8,
    min_samples_for_NMESC: int = 6,
    enhanced_count_thres: int = 80,
    max_rp_threshold: float = 0.15,
    sparse_search_volume: int = 30,
    fixed_thres: float = 0.0,
    cuda=False,
):
    """
    Clustering method for speaker diarization based on cosine similarity.
    NME-SC part is converted to torch.tensor based operations in NeMo 1.9.

    Args:
        uniq_embs_and_timestamps: (dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_embs_and_timestamps contains only one scale, single scale diarization
            is performed.

        oracle_num_speaker: (int or None)
            The oracle number of speakers if known else None

        max_num_speaker: (int)
            The maximum number of clusters to consider for each session

        min_samples_for_NMESC: (int)
            The minimum number of samples required for NME clustering. This avoids
            zero p_neighbour_lists. If the input has fewer segments than min_samples,
            it is directed to the enhanced speaker counting mode.

        enhanced_count_thres: (int)
            For the short audio recordings under 60 seconds, clustering algorithm cannot
            accumulate enough amount of speaker profile for each cluster.
            Thus, getEnhancedSpeakerCount() employs anchor embeddings (dummy representations)
            to mitigate the effect of cluster sparsity.
            enhanced_count_thres = 80 is recommended.

        max_rp_threshold: (float)
            Limits the range of parameter search.
            Clustering performance can vary depending on this range.
            Default is 0.15.

        sparse_search_volume: (int)
            Number of p_values we search during NME analysis.
            Default is 30. The lower the value, the faster NME-analysis becomes.
            Lower than 20 might cause a poor parameter estimation.

        fixed_thres: (float)
            If fixed_thres value is provided, NME-analysis process will be skipped.
            This value should be optimized on a development set to obtain a quality result.
            Default is None and performs NME-analysis to estimate the threshold.

    Returns:
        Y: (torch.tensor[int])
            Speaker label for each segment.
    """
    device = torch.device("cuda") if cuda else torch.device("cpu")

    # Get base-scale (the highest index) information from uniq_embs_and_timestamps.
    uniq_scale_dict = uniq_embs_and_timestamps['scale_dict']
    emb = uniq_scale_dict[max(uniq_scale_dict.keys())]['embeddings']

    if emb.shape[0] == 1:
        return torch.zeros((1,), dtype=torch.int32)
    elif emb.shape[0] <= max(enhanced_count_thres, min_samples_for_NMESC) and oracle_num_speakers is None:
        est_num_of_spk_enhanced = getEnhancedSpeakerCount(emb, cuda)
    else:
        est_num_of_spk_enhanced = None

    if oracle_num_speakers:
        max_num_speaker = oracle_num_speakers

    mat, emb = getMultiScaleCosAffinityMatrix(uniq_embs_and_timestamps, device)

    nmesc = NMESC(
        mat,
        max_num_speaker=max_num_speaker,
        max_rp_threshold=max_rp_threshold,
        sparse_search=True,
        sparse_search_volume=sparse_search_volume,
        fixed_thres=fixed_thres,
        NME_mat_size=300,
        cuda=cuda,
        device=device,
    )

    if emb.shape[0] > min_samples_for_NMESC:
        est_num_of_spk, p_hat_value = nmesc.NMEanalysis()
        affinity_mat = getAffinityGraphMat(mat, p_hat_value)
    else:
        affinity_mat = mat

    if oracle_num_speakers:
        est_num_of_spk = oracle_num_speakers
    elif est_num_of_spk_enhanced:
        est_num_of_spk = est_num_of_spk_enhanced

    spectral_model = SpectralClustering(n_clusters=est_num_of_spk, cuda=cuda, device=device)
    Y = spectral_model.predict(affinity_mat)

    return Y.cpu().numpy()
