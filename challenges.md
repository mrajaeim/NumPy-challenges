## 🔹 Medium Level

### 1. Block-wise Mean Reduction

Given a 2D array of shape `(H, W)` where both are divisible by `k`, compute the mean of each `k × k` block **without loops**.

---

### 2. Row-wise Z-score Normalization

Normalize each row of a 2D array using Z-score, handling zero-variance rows safely.

---

### 3. Efficient One-Hot Encoding

Given an integer array of labels `(n,)`, create a one-hot matrix `(n, num_classes)` without Python loops.

---

### 4. Pairwise Euclidean Distance Matrix

Given matrix `X (n, d)`, compute full pairwise Euclidean distance matrix `(n, n)` using broadcasting (no loops).

---

### 5. Sliding Window View (Manual)

Given 1D array and window size `k`, create a 2D array of all sliding windows using `np.lib.stride_tricks.as_strided`.

---

### 6. Top-k Indices Per Row

Given `(n, m)` matrix, return indices of top `k` values per row efficiently (without full sorting).

---

### 7. Stable Softmax

Implement numerically stable softmax for 2D array (row-wise).

---

### 8. Covariance Matrix (From Scratch)

Given data matrix `(n_samples, n_features)`, compute covariance matrix manually (without `np.cov`).

---

### 9. Detect Duplicate Rows

Find all duplicate rows in a large 2D array efficiently.

---

### 10. Fast Histogram Without `np.histogram`

Implement histogram calculation using `np.bincount`.

---

## 🔹 Medium-Hard Level

### 11. Batch Matrix Multiplication

Given arrays `(b, n, m)` and `(b, m, p)`, compute batch matrix multiplication without loops.

---

### 12. Vectorized Haversine Distance

Given latitude/longitude arrays `(n, 2)`, compute pairwise spherical distances.

---

### 13. Rolling Mean (2D Image)

Implement 2D moving average filter using only NumPy (no OpenCV/Scipy).

---

### 14. Reconstruct Matrix from Diagonals

Given main diagonal and k-th diagonals, reconstruct matrix efficiently.

---

### 15. Efficient Boolean Mask Update

Given condition on array, replace elements conditionally **without branching logic**.

---

### 16. Gram Matrix Computation

Given `X (n, d)`, compute Gram matrix `G = X @ X.T` and normalize it.

---

### 17. Mahalanobis Distance

Compute Mahalanobis distance for each row given covariance matrix.

---

### 18. Vectorized Polynomial Feature Expansion

Given `(n, d)` input, generate polynomial features up to degree 3 without loops.

---

## 🔹 Hard Level

### 19. PCA From Scratch

Implement PCA:

* Center data
* Compute covariance
* Eigen decomposition
* Project to k dimensions

Without sklearn.

---

### 20. K-Means (Fully Vectorized)

Implement one full K-means iteration (assignment + centroid update) without explicit loops.

---

### 21. Fast Non-Maximum Suppression

Given bounding boxes `(n,4)` and scores `(n,)`, implement NMS in NumPy.

---

### 22. Memory-Efficient Large Matrix Multiplication

Multiply very large matrices in chunks to avoid memory overflow.

---

### 23. Implement Convolution (Multi-channel)

Implement 2D convolution for multi-channel image and kernel manually using NumPy.

---

### 24. Batched Log-Sum-Exp

Implement numerically stable batched log-sum-exp.

---

### 25. Neural Network Forward Pass (Fully Vectorized)

Given:

* `X (n, d)`
* `W1, b1`
* `W2, b2`

Implement:

```
hidden = relu(X @ W1 + b1)
output = softmax(hidden @ W2 + b2)
```

Fully vectorized and numerically stable.