# Dictionary-Learning

 

In recent years, matrix-factorization methods have become ubiquitous in machine learning, and have been used in a wide range of applications such as denoising, inpainting, recommender systems or clustering. This called for more scalable and robust methods to represent a signal, and recent advances in dictionary learning aim at learning both the decomposition of a signal in a basis and the basis itself.

A signal $\mathbf{x}_i \in \mathbb{R}^{N}$ is decomposed in a basis (called the dictionary) $\mathbf{D} = [\mathbf{d}_1, \dots, \mathbf{d}_k ] \in \mathbb{R}^{N\times k}$, such that $\mathbf{x}_i \approx \mathbf{D} \boldsymbol{\alpha}_i$ (in the $\ell_2$ sense). The columns $\mathbf{d}_i, i=1, \dots, k$ of the dictionary are often referred to as the atoms and $\boldsymbol{\alpha}_i \in \mathbb{R}^{k}$ represent the weight of each of these in the representation of $\mathbf{x}_i$.

The empirical risk minimization for a dataset with samples $\mathbf{x}_i, \ i=1,\dots,n$ reads :


$$
\underset{\mathbf{D} \in \mathcal{C}, \ \alpha \in \mathbb{R}^{k \times n}}{\min} \ \dfrac{1}{n} \sum_{i=1}^n (\dfrac{1}{2} \lVert\mathbf{x}_i - \mathbf{D} \boldsymbol{\alpha}_i \lVert_2^2 \ + \ \lambda \lVert\boldsymbol{\alpha}_i\lVert_1)
$$

where $\lambda>0$ is a regularization parameter enforcing sparsity, $\mathcal{C} = \{ \mathbf{D} \in \mathbb{R}^{N \times k} \ \text{s.t.} \ \forall j = 1,...,k; \ \mathbf{d}_j^T \mathbf{d}_j \leq 1 \}$, so that the $\boldsymbol{\alpha}_i$ are not arbitrarily small. This function is minimized by alternatively optimizing the dictionary $\mathbf{D}$ and the parameters $\boldsymbol{\alpha}$.

Further advances 

"...the **go to** statement should be abolished..." [[1]](#1).

## References
<a id="1">[1]</a> 
Julien Mairal, Jean Ponce, Guillermo Sapiro, Andrew Zisserman, and Francis Bach (2008). Supervised dictionary learning. Advances in neural information processing systems, 21. https://papers.nips.cc/paper/2008/file/c0f168ce8900fa56e57789e2a2f2c9d0-Paper.pdf

<a id="2">[2]</a> 
Julien Mairal, Francis Bach, Jean Ponce, and Guillermo Sapiro (2009). Online dictionary learning for sparse coding. Proceedings of the 26th Annual International Conference on Machine Learning. https://doi.org/10.1145/1553374.1553463


      

