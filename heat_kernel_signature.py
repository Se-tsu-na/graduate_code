"""This module implements Heat Kernel and Heat Kernel Signature for a mesh."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import trimesh


def sqr_euc_dis(vertices):
    """Calculates the squared distances between vertices in a mesh.平方距离

    Args:
      vertices: A trimesh.caching.TrackedArray of shape '[n, d]'.
        n is the number of vertices in the mesh and d is the number of
        dimensions.

    Returns：
      Tensor with shape '[n, n]'
    """
    dot_mat = tf.matmul(vertices, tf.transpose(vertices))  #矩阵乘法

    sqr_norm_mat = tf.reshape(  #重构
        tf.square(  #元素平方
            tf.norm(  #范数
                vertices, axis=1)), [
            len(vertices), 1])
    sqr_norm_mat = tf.tile(sqr_norm_mat, [1, len(vertices)])  #维度上的复制

    sqr_euc_dis = sqr_norm_mat + tf.transpose(sqr_norm_mat) - 2 * dot_mat  #转置

    return sqr_euc_dis


def compute_Dinv(vertices, t):
    """Calculates the inverse of D.

    D is a diagonal matrix used in the approximation for the Mesh Laplacian,
      L:
        L = I + Dinv*H

    Args:
      vertices: A trimesh.caching.TrackedArray of shape '[n, d]'.
        n is the number of vertices in the mesh and d is the number of
        dimensions.
      t: A float.

    Returns：
      A "SparseTensor" with shape '[n, n]'
    """
    num = len(vertices)
    Euc_Dis = sqr_euc_dis(vertices)

    diagonal = tf.math.reciprocal_no_nan(  #元素倒数
        tf.math.reduce_sum(tf.math.exp(-tf.math.truediv(Euc_Dis, t)), 1)
        # 用于查找张量的各维元素之和（降维求和）axis=0按列，1按行
        #tensor逐个元素e^
        #tensor按元素逐个除法
    )
    idx = [[i, i] for i in range(num)]

    return tf.sparse.SparseTensor(  #稀疏矩阵
        indices=idx,  #非零值索引
        values=diagonal,  #非零值tensor
        dense_shape=[
            num,
            num])  #shape
    #返回值是个对角矩阵，（i，i）号元素是 e^(d(i,j)*d(i,j)/t)对j求和


def compute_H(vertices, t):
    """Calculates H.

    H is a matrix used in the approximation for the Mesh Laplacian, L:
        L = I + Dinv*H

    Args:
      vertices: A trimesh.caching.TrackedArray of shape '[n, d]'.
        n is the number of vertices in the mesh and d is the number of
        dimensions.
        n行3列，每一行为三维模型的一个顶点的三维坐标，共n个顶点
      t: A float.

    Returns：
      Tensor with shape '[n, n]'
    """
    num = len(vertices)
    Euc_Dis = sqr_euc_dis(vertices)

    H = tf.math.exp(-tf.math.truediv(Euc_Dis, t))

    return H
    # 返回值是个矩阵，（i，j）号元素是 e^(d(i,j)*d(i,j)/t)

    #所以这里Dinv*H到底是个什么东西


def mesh_Laplacian(vertices, t):
    """Calculates the mesh Laplacian matrix for a mesh.

    The mesh Laplacian matrix, L is approximated as follows:
      L = I + Dinv*H

    Args:
      vertices: A trimesh.caching.TrackedArray of shape '[n, d]'.
        n is the number of vertices in the mesh and d is the number of
        dimensions.
      t: A float.

    Returns：
      Tensor with shape '[n, n]'
    """
    num = len(vertices)
    I = tf.sparse.eye(num, dtype=tf.dtypes.float64)  # sparse
    Dinv = compute_Dinv(vertices, t)  # sparse
    H = compute_H(vertices, t)  # dense

    L = tf.sparse.add(
        I,
        -tf.linalg.matmul(
            tf.sparse.to_dense(Dinv), H, a_is_sparse=True, b_is_sparse=False
        ),
    )

    return L


def eig_dec(L):
    """Performs eigen-decomposition on a matrix.

    Args:
      L： Tensor with shape '[n, n]'

    Returns：
      eig_val: A Tensor with shape '[n, ]'. Eigenvalues are in increasing order.
      eig_vec: A Tensor with shape '[n, n]'. Columns are eigenvectors.
    """
    eig_val, eig_vec = tf.linalg.eigh(L)
    return eig_val, eig_vec


def heat_kernel(vertices, x, t):
    """Calculates Heat Kernel at x.

    Args:
      vertices: A trimesh.caching.TrackedArray of shape '[n, d]'.
        n is the number of vertices in the mesh and d is the number of
        dimensions.
      x: An int. It is the index of the vertex to calculate the heat kernel.
      t: A float.

    Returns：
      Tensor with shape '[n, ]'
    """
    L = mesh_Laplacian(vertices, t)
    eig_val, eig_vec = eig_dec(L)

    # compute k
    eterm = tf.exp(-eig_val * t)
    kxy_t = tf.math.reduce_sum(eterm * eig_vec[x] * eig_vec, axis=1)

    return kxy_t


def hks_t(vertices, t):
    """Calculates Heat Kernel Signature k(x,x) for all vertices with a t.

    Args:
      vertices: A trimesh.caching.TrackedArray of shape '[n, d]'.
        n is the number of vertices in the mesh and d is the number of
        dimensions.
      t: A float.

    Returns：
      Tensor with shape '[n, ]'
    """
    L = mesh_Laplacian(vertices, t)
    eig_val, eig_vec = eig_dec(L)

    # compute k
    eterm = tf.exp(-eig_val * t)
    k_t = tf.math.reduce_sum(eterm * (eig_vec ** 2), axis=1)

    return k_t


def HKS(vertices, ts):
    """Calculates Heat Kernel Signature k(x,x) for all vertices with a few t's.

    Args:
      vertices: A trimesh.caching.TrackedArray of shape '[n, d]'.
        n is the number of vertices in the mesh and d is the number of
        dimensions.
      ts: A list of floats of length m

    Returns：
      Tensor with shape '[n, m]'
    """

    num = len(vertices)
    K = np.ones((num, len(ts)))
    for idx, t in enumerate(sorted(ts)):
        K[:, idx] = hks_t(vertices, t)

    return tf.constant(K)
