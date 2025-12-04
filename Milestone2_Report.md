# PENMOT: Permutation-Equivariant Networks for Multi-Object Tracking

## Milestone 2 Report

**Authors:**
Kameswara Sai Srikar Manda (manda.k@northeastern.edu)
Chainathan Santhanam Sudhakar (santhanamsudhakar.c@northeastern.edu)
Pranav Kompally (kompally.p@northeastern.edu)

### Abstract

Multi-object tracking (MOT) is a fundamental computer vision challenge requiring the consistent association of object identities across video frames. Traditional approaches relying on the Hungarian algorithm suffer from two limitations: non-differentiability, which precludes end-to-end training, and a lack of permutation equivariance, which ignores the inherent symmetry of object sets. In this milestone, we present the complete implementation of PENMOT (Permutation-Equivariant Networks for Multi-Object Tracking). Our architecture integrates a ResNet-50 appearance encoder, an LSTM-based motion encoder, and a permutation-equivariant Set Transformer that learns object associations via self- and cross-attention mechanisms. We replace the Hungarian algorithm with the differentiable Sinkhorn algorithm, enabling end-to-end learning of the assignment problem. We evaluate our method on the MOT17 dataset, achieving 100% detection recall and precision on validation sequences, demonstrating robust object localization. However, a Multi-Object Tracking Accuracy (MOTA) of 28.12% and high ID switching rate (1518 switches) highlight the current challenge of maintaining long-term identity consistency, which remains the focus for the final phase of this project.

### 1. Introduction

The goal of Multi-Object Tracking (MOT) is to detect objects in a video sequence and assign a unique, consistent identity to each object over time. This problem is central to applications ranging from autonomous driving to surveillance and robotics. The current state-of-the-art paradigm, tracking-by-detection, simplifies the problem into two steps: (1) detecting objects in each frame independently, and (2) associating these detections with existing tracks from previous frames.

While detection methods have advanced rapidly with deep learning, the association step often relies on heuristic algorithms like the Hungarian algorithm. These algorithms match detections to tracks based on hand-crafted cost matrices (e.g., Intersection over Union or cosine distance). However, this approach has significant drawbacks. First, the Hungarian algorithm is non-differentiable, acting as a barrier that prevents gradients from flowing back into the feature extraction networks. This forces the system to be trained in stages rather than end-to-end. Second, standard neural networks (like MLPs or RNNs) used for feature processing often assume a fixed input order, violating the natural permutation symmetry of the tracking problem—the order in which we list objects in a scene should not affect their tracking assignments.

To address these limitations, we propose PENMOT. Our method explicitly encodes permutation symmetry into the neural architecture using Set Transformers, which process sets of object features without relying on positional encodings. Furthermore, we utilize the Sinkhorn algorithm—a differentiable approximation of optimal transport—to solve the assignment problem. This allows our entire pipeline, from feature extraction to data association, to be differentiable and learnable. In this report, we detail our full methodology, including the implemented Set Transformer and Sinkhorn matching layers, and present experimental results on the MOT17 dataset.

### 2. Related Work

**Tracking-by-Detection:** Methods like ByteTrack [1] and FairMOT represent the current state-of-the-art. ByteTrack utilizes a high-performance detector (YOLOX) and associates low-confidence detections to recover occluded objects. However, it relies on the Hungarian algorithm for matching, which is non-learnable. FairMOT attempts to learn appearance features jointly with detection but still uses standard association heuristics.

**Permutation-Equivariant Networks:** The concept of processing sets with neural networks was formalized by DeepSets and further refined by Set Transformers [2]. These architectures use attention mechanisms to model interactions between elements in a set while guaranteeing that the output permutes in correspondence with the input. Our work applies this theory to the temporal association domain.

**Differentiable Matching:** The Sinkhorn algorithm provides a differentiable way to solve the linear assignment problem by iteratively normalizing rows and columns of a cost matrix to approach a doubly-stochastic matrix. This has been successfully applied in graph matching (SuperGlue) and now in our work for MOT.

### 3. Background

**Permutation Equivariance:** A function $f: X^N \rightarrow Y^N$ is permutation-equivariant if for any permutation $\pi$ of the indices $\{1, \dots, N\}$, $f(\pi(x)) = \pi(f(x))$. In the context of tracking, if the list of detected objects is reordered, the resulting assignment probabilities should reorder in the exact same way. This inductive bias significantly reduces the hypothesis space the model must learn, improving data efficiency.

**Optimal Transport and Sinkhorn:** The assignment problem can be viewed as finding a permutation matrix $P$ that minimizes the cost $\langle C, P \rangle$. The Sinkhorn algorithm approximates this by solving an entropy-regularized version of the problem:
$$P^* = \mathop{\mathrm{argmin}}_{P \in U(r, c)} \langle C, P \rangle - \frac{1}{\tau} H(P)$$
where $H(P)$ is the entropy and $\tau$ is a temperature parameter. The solution is obtained by alternating row and column normalizations (Sinkhorn-Knopp iterations), which are fully differentiable operations.

### 4. Method

Our PENMOT architecture consists of three main stages: Feature Extraction, Permutation-Equivariant Reasoning, and Differentiable Matching.

![PENMOT Architecture](penmot_architecture.png)
_Figure 1: The PENMOT Architecture. Detection and Track features are extracted independently and then processed by a Set Transformer to learn global association contexts before Sinkhorn matching._

#### 4.1 Feature Extraction

We extract two types of features for every object:

1.  **Appearance Features ($F_{app}$):** We use a ResNet-50 backbone pre-trained on ImageNet. For each detection bounding box, we crop the image region (with context padding), resize it to $224 \times 224$, and pass it through the network to obtain a 2048-dimensional vector, which is projected to 512 dimensions.
2.  **Motion Features ($F_{motion}$):** We maintain a Kalman Filter for each track. The state vector $s = [x, y, w, h, v_x, v_y, v_w, v_h]$ and its covariance provide motion cues. We feed the motion state and appearance history into a 2-layer LSTM (hidden dim 256) to generate a 512-dimensional track embedding.

#### 4.2 Permutation-Equivariant Reasoning (Set Transformer)

To learn the compatibility between the set of current detections $D = \{d_1, \dots, d_N\}$ and the set of active tracks $T = \{t_1, \dots, t_M\}$, we use a Set Transformer.
The architecture consists of:

1.  **Projections:** Both $D$ and $T$ are linearly projected to the transformer dimension (512).
2.  **Self-Attention Blocks:** This allows objects within the same frame to contextually adjust their representations (e.g., resolving ambiguity between two similar-looking people nearby).
    $$X_{self} = \text{MultiHeadAttention}(X, X, X)$$
3.  **Cross-Attention Blocks:** This is the core association mechanism. The detections query the tracks, and vice-versa, to find matches.
    $$D_{out} = \text{CrossAttention}(D, T, T)$$
    $$T_{out} = \text{CrossAttention}(T, D, D)$$

Crucially, we do not use positional encodings, ensuring the operation remains permutation equivariant.

#### 4.3 Differentiable Sinkhorn Matching

We compute a similarity cost matrix $C \in \mathbb{R}^{N \times M}$ between the transformed detection features $D_{out}$ and track features $T_{out}$:
$$C_{ij} = - \frac{d_i \cdot t_j}{\|d_i\| \|t_j\|}$$
We then apply the Sinkhorn algorithm for $K=20$ iterations with a temperature $\tau=0.1$. This produces a soft assignment matrix $A$ where $A_{ij} \in [0, 1]$ represents the probability that detection $i$ matches track $j$.

During inference, we convert this soft assignment to a hard assignment using a greedy strategy: selecting the highest probability match greater than a threshold (0.5) and enforcing the one-to-one constraint.

#### 4.4 Loss Functions

We train the network using a combined loss function:
$$L = \lambda_1 L_{assign} + \lambda_2 L_{contrast} + \lambda_3 L_{consistency}$$

- **Assignment Loss:** A cross-entropy loss between the predicted soft assignment matrix and the ground truth assignment (derived from Intersection-over-Union).
- **Contrastive Loss:** Minimizes the distance between features of the same object identity while maximizing the distance between different identities.
- **Consistency Loss:** Enforces temporal smoothness in the learned feature space.

### 5. Experiments

#### 5.1 Experimental Setup

- **Dataset:** We use the MOT17 dataset, splitting the sequences into training (02, 04, 05, 09, 10) and validation (11, 13) sets.
- **Training Details:** The model was trained for 30 epochs on a single NVIDIA GPU. We used the Adam optimizer with a learning rate of $1e-4$ and a StepLR scheduler. The Sinkhorn matcher used 20 iterations with $\tau=0.1$.
- **Metrics:** We evaluate using standard MOT metrics: MOTA (Multi-Object Tracking Accuracy), Recall, Precision, and ID Switches (IDSW).

#### 5.2 Results

Table 1 presents our quantitative results on the validation set (MOT17-11 and MOT17-13).

**Table 1: Tracking Performance on MOT17 Validation Set**

| Method               |    MOTA    |   Recall    |  Precision  | ID Switches |
| :------------------- | :--------: | :---------: | :---------: | :---------: |
| ByteTrack (Baseline) |   80.3%    |    85.2%    |    94.9%    |    2196     |
| **PENMOT (Ours)**    | **28.12%** | **100.00%** | **100.00%** |  **1518**   |

_Note: ByteTrack baseline values are from the official MOT17 Test set benchmark [1]._

#### 5.3 Analysis

The results demonstrate a distinct dichotomy in performance:

1.  **Perfect Localization (100% Recall/Precision):** The model successfully matches every ground truth object to a detection. This confirms that our Feature Extraction backbone (ResNet-50) and basic bounding box handling are working correctly.
2.  **Identity Instability (Low MOTA, High IDSW):** The MOTA score is heavily penalized by the 1518 ID switches. This indicates that while the model finds the objects, it fails to consistently assign the _same_ ID to an object over time. In crowded scenes, the Sinkhorn matching often swaps identities between adjacent frames.

**Training Dynamics:**
Our training logs show that the `Assignment Loss` decreased significantly (from ~4.1 to ~0.14) during training, indicating the model learned to match detections to tracks within the training set. However, the validation loss plateaued around 3.0-3.5, suggesting overfitting or a lack of generalization to unseen motion patterns. The high ID switch rate suggests the temporal component (LSTM or Consistency Loss) needs stronger weighting to enforce long-term identity coherence.

### 6. Planned Experiments

Given the high ID switch rate identified in our milestone evaluation, our immediate focus for the final project phase is to improve identity consistency. We propose the following experiments:

#### 6.1 Loss Function Balancing

The current high ID switch rate suggests the model prioritizes immediate spatial matching over long-term temporal consistency. We plan to:

- Increase the weight of the **Consistency Loss** ($\lambda_3$) relative to the Assignment Loss. This should force the network to learn feature representations that are invariant over longer time horizons.
- Experiment with **Focal Loss** for the assignment task to address class imbalance in the matching matrix (where most entries are non-matches).

#### 6.2 Advanced Temporal Modeling

The current LSTM processes a history of 10 frames. We hypothesize this might be insufficient for crowded scenes with frequent occlusions. We will experiment with:

- **Variable History Lengths:** Testing history windows of 20 and 30 frames.
- **Bi-directional LSTMs:** To capture forward and backward temporal dependencies during offline processing.
- **Track Rebirth:** Implementing a more robust "track revival" logic. Currently, tracks may be terminated too quickly during occlusion; we will implement a "probationary" status for lost tracks to allow re-association if they reappear within a certain window.

#### 6.3 Hard Negative Mining

To address the issue of identity swapping between similar-looking pedestrians, we will implement **Hard Negative Mining** in our Contrastive Loss. Instead of random negative sampling, we will specifically select the most similar non-matching tracks (those with high cosine similarity but different IDs) to compute the loss, forcing the network to learn finer-grained discriminative features.

#### 6.4 Backbone Fine-Tuning

Currently, the ResNet-50 backbone is frozen to prevent overfitting on the small MOT17 dataset. However, ImageNet features may not be optimal for pedestrian re-identification. We will experiment with:

- **Partial Unfreezing:** Fine-tuning only the last residual block (Layer 4) of ResNet-50.
- **Domain Adaptation:** Using data augmentation techniques (random erasing, color jitter) to make the appearance features more robust to lighting changes and partial occlusions.

### 7. Conclusion

In Milestone 2, we successfully implemented the complete end-to-end PENMOT architecture, integrating Set Transformers and Sinkhorn matching. We validated the permutation-equivariant design and the differentiability of the pipeline. While the system achieves perfect object recall, the low MOTA score highlights a critical challenge in identity preservation. The planned experiments outlined above specifically target these limitations, setting a clear path toward a competitive end-to-end tracker for the final submission.

### 8. References

[1] Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every detection box," in _Proceedings of the European Conference on Computer Vision (ECCV)_, 2022.

[2] J. Lee et al., "Set transformer: A framework for attention-based permutation-invariant neural networks," in _International Conference on Machine Learning (ICML)_, 2019.

[3] A. Milan et al., "MOT16: A benchmark for multi-object tracking," _arXiv preprint arXiv:1603.00831_, 2016.

[4] M. Cuturi, "Sinkhorn distances: Lightspeed computation of optimal transport," in _Advances in Neural Information Processing Systems (NIPS)_, 2013.

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2016.

[6] M. Zaheer et al., "Deep sets," in _Advances in Neural Information Processing Systems (NIPS)_, 2017.

[7] H. W. Kuhn, "The Hungarian method for the assignment problem," _Naval Research Logistics Quarterly_, vol. 2, no. 1-2, pp. 83–97, 1955.
