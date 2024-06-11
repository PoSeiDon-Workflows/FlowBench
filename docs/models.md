
* PyOD:

```python
classifiers = {
  'Angle-based Outlier Detector (ABOD)':
      ABOD(),
  'Cluster-based Local Outlier Factor (CBLOF)':
      CBLOF(check_estimator=False, random_state=random_state, n_jobs=-1),
  'Feature Bagging':
      FeatureBagging(LOF(n_neighbors=35), random_state=random_state, n_jobs=-1),
  'Histogram-base Outlier Detection (HBOS)':
      HBOS(),
  'Isolation Forest':
      IForest(random_state=random_state, n_jobs=-1),
  'K Nearest Neighbors (KNN)':
      KNN(n_jobs=-1),
  'Average KNN (AKNN)':
      KNN(method='mean', n_jobs=-1),
  'Local Outlier Factor (LOF)':
      LOF(n_neighbors=35, n_jobs=-1),
  'Minimum Covariance Determinant (MCD)':
      MCD(random_state=random_state),
  'One-class SVM (OCSVM)':
      OCSVM(),
  'Principal Component Analysis (PCA)':
      PCA(random_state=random_state),
  'Locally Selective Combination (LSCP)':
      LSCP(detector_list, random_state=random_state),
  'Isolation-based anomaly detection using nearest-neighbor ensembles (INNE)':
      INNE(max_samples=2, random_state=random_state),
  'Gaussian Mixture Model (GMM)':
      GMM(random_state=random_state),
  'Kernel Density Estimation (KDE)':
      KDE(),
  'Linear Method for Deviation-based Outlier Detection (LMDD)':
      LMDD(random_state=random_state),
}
```