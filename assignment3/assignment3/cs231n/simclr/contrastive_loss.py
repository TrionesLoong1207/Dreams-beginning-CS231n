import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.

    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.


    中文对照：
    计算两个 1×D 特征向量 z_i、z_j 的归一化点积，即余弦相似度，返回标量。
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    # [中文] 待完成。原题提示：可参考 torch.linalg.norm。
    i_norm = torch.linalg.norm(z_i)
    j_norm = torch.linalg.norm(z_j)
    z_i = z_i.reshape(-1)
    z_j = z_j.reshape(-1)
    
    norm_dot_product = z_i.dot(z_j) / (i_norm * j_norm)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 自行实现的代码区域结束。
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).

    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair.
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.

    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.


    中文对照：
    通过循环计算一个批量的 SimCLR 对比损失。out_left、out_right 均为 (N,D)，表示两个增强分支经投影头后的特征；temperature 为温度参数。返回标量批量损失。
    """
    N = out_left.shape[0]  # total number of training examples
    # [中文] 训练样本总数。
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
     # [中文] 将 out_left 与 out_right 拼成形状为 (2N,D) 的张量。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
    # [中文] 遍历各个正样本对 (k,k+N)。
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # [中文] 待完成。原题提示：计算 l(k,k+N) 和 l(k+N,k)。
        e = torch.exp(sim(z_k, z_k_N)/tau)
        e_sum = 0
        for m in range(2*N):
            if m != k:
                e_sum += torch.exp(sim(z_k, out[m])/tau)
        L_k = -torch.log(e / e_sum)

        e_sum = 0
        e = torch.exp(sim(z_k_N, z_k)/tau)
        for m in range(2*N):
            if m != k + N:
                e_sum += torch.exp(sim(z_k_N, out[m])/tau)
        L_k_N = -torch.log(e / e_sum)

        loss = L_k + L_k_N
        total_loss += loss
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # [中文] 自行实现的代码区域结束。
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    # [中文] 最后将总损失除以批内增强样本总数 2N。
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.

    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].


    中文对照：
    计算两个增强分支中对应正样本对的归一化点积。out_left、out_right 为 (N,D)，每对由同一原图的两个增强视图构成；输出对应的相似度。
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    # [中文] 待完成。原题提示：可参考 torch.linalg.norm。
    pos_pairs = torch.sum(out_left * out_right, dim=1, keepdim=True)
    l_norm = torch.linalg.norm(out_left, dim=1, keepdim=True)
    r_norm = torch.linalg.norm(out_right, dim=1, keepdim=True)

    pos_pairs /= l_norm * r_norm
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 自行实现的代码区域结束。
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.

    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].


    中文对照：
    计算批内 2N 个增强样本两两之间的归一化点积矩阵。输入 out 为 (2N,D)，每行是一个投影特征；输出为 (2N,2N)。
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################
    # [中文] 待完成：从这里开始编写代码。
    norm = torch.linalg.norm(out, dim=1, keepdim=True)
    out_normalized = out / norm
    sim_matrix = out_normalized @ out_normalized.T
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 自行实现的代码区域结束。
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.

    Inputs and output are the same as in simclr_loss_naive.


    中文对照：
    用向量化方式计算 SimCLR 批量对比损失，不允许显式循环。输入输出与 simclr_loss_naive 相同。
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    # [中文] 将 out_left 与 out_right 拼成形状为 (2N,D) 的张量。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    # [中文] 计算批内所有增强样本两两之间的相似度矩阵。
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    # [中文] 待完成：按原题提示编写代码。
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    # [中文] 步骤 1：根据 sim_matrix 计算所有增强样本的分母。原题提示：先计算 exp(sim/tau)，保存为形状 2N×2N 的 exponential。
    exponential = None
    exponential = torch.exp(sim_matrix/tau)
    # This binary mask zeros out terms where k=i.
    # [中文] 此二值掩码将 k=i 的项置零。
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    # [中文] 应用二值掩码。
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    # [中文] 原题提示：对所有增强样本计算分母，结果应为形状 2N×1 的向量。
    denom = None
    denom = torch.sum(exponential, dim=1, keepdim=True)
    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    # [中文] 步骤 2：计算正样本对的相似度。可以从 sim_matrix 提取对应元素，也可以调用 sim_positive_pairs()。
    pos_pairs = sim_positive_pairs(out_left, out_right)
    
    # Step 3: Compute the numerator value for all augmented samples.
    # [中文] 步骤 3：计算所有增强样本的分子。
    numerator = None
    numerator = torch.exp(pos_pairs/tau)
    numerator = torch.cat([numerator, numerator], dim=0)
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    # [中文] 步骤 4：已有所有增强样本的分子与分母后，计算总损失。
    loss = None
    loss = -torch.log(numerator/denom).mean()
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 自行实现的代码区域结束。
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))