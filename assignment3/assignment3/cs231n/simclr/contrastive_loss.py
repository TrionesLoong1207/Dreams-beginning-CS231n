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
    两种向量之间的正常点产品。

    输入数: - z_i: 1xD 发热。 - z_j: 1 xD 发光。

    返回: - 一个标量值,即z_i 和 z_j 之间的正统点产值。
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    # [中文] TODO: 启动您的代码。 HINT: torch.linalg.norm 可能会有帮助。
    
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 代码结束
    
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
    计算批量 (naive loop version) 的对比性损失L。

    输入: - out_left: NxD shor; 投影头 g () 的输出, SimCLR 模型中的左分支。 out_right: nxD Shor; 预测头 G () 输出, SIMCL 模型中的右分支。 每个行都是批量中放大样本的 Z- Vector。 在 out_left 和 out_right 中的同一行形成正对。 换句话说, ( out_left, out_right [k]) 对所有 k = 0 ... N- 1 形成正对。 - Tau: 标值, 温度参数决定指数增长的速度。

    返回: - 星标值;批量中所有正对的总损失。定义见笔记本。
    """
    N = out_left.shape[0]  # total number of training examples
    # [中文] 训练实例总数
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
     # [中文] 将 out_left 和 out_right 相连接为 2 * N x D 发光器。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
    # [中文] 环绕每对正对( k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # [中文] TODO: 代码启动。 提示: 计算 l( k, k+N) 和 l( k+N, k)。
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # [中文] 代码结束
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    # [中文] 最后,我们需要将损失总数除以2N,即批量样品的数量。
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
    正样本对之间的归一化点积。

    输入: - out_left: NxD shor; 投影头 g () 的输出, SimCLR 模型中的左分支。 - out_right: N xD shorn; 投像头 g() 的产出, SimClR 模型中右分支。 每行都是批量中放大样本的 Z 矢量。 同一行 ( out_left ) 和 out_right 形成正对数。

    返回: - Nx1 发压器;每行 k 是 out_left [k] 和 out_right [k]之间的正统点产物。
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    # [中文] TODO: 启动您的代码。 HINT: torch.linalg.norm 可能会有帮助。
    
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 代码结束
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.

    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].


    中文对照：
    在一个批量中计算所有配对增强型示例之间的 2N x 2N 归一化点积矩阵。

    输入: - 输出: 2N x D 发压器; 每行为单个增强示例的 z 矢量 (output of projection head)。 批量中共有 2N 增强示例。

    返回: - sim_matrix: 2N x 2N 温度;矩阵中每个元素 i, j 是 out[i] 和 out[j] 之间的正统点产物。
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################
    # [中文] 开始你的代码。
    

    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 代码结束
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.

    Inputs and output are the same as in simclr_loss_naive.


    中文对照：
    计算批量 (vectorized version) 的对比性损失 L。 不允许循环。

    输入和输出与 simclr_loss_naive 相同。
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    # [中文] 将 out_left 和 out_right 相连接为 2 * N x D 发光器。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    # [中文] 计算批量中所有增加示例的相似性矩阵。
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    # [中文] 启动您的代码。 跟随提示。
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    # [中文] 步骤1:使用sim_matrix来计算所有增强样本的分母值。提示:计算 esim / tau} 并存储成指数值, 该指数值应具有 2N x 2N 形状。
    exponential = None
    
    # This binary mask zeros out terms where k=i.
    # [中文] 此二进制掩码在 k=i 处为零。
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    # [中文] 我们用二进制面具
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    # [中文] 提示: 计算所有增强样本的分母值。 这应该是 2N x 1 矢量。
    denom = None

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    # [中文] 步骤 2: 计算正对之间的相似性。 您可以以两种方式做到这一点: 选项 1: 从 sim_matrix 中提取相应的索引。 选项 2: 使用
    # sim_positive_pairs ()。
    
    
    # Step 3: Compute the numerator value for all augmented samples.
    # [中文] 步骤3:计算所有增强样品的分子值。
    numerator = None
    
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    # [中文] 步骤4:现在您拥有所有增强样品的分子和分母,计算总损失。
    loss = None
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 代码结束
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))