# svd-and-cur-recommenders

SVD and CUR algorithms implemented using Python

<!-- A table -->

| Algorithm             | RMSE      | Top K Precision (K = 5) | Spearman Correlation Coefficient | Time |
| --------------------- | --------- | ----------------------- | -------------------------------- | ---- |
| SVD                   | 2.052e-15 | 81.0638                 | 0.956                            | ~2s  |
| SVD with 90% variance | 0.523     | 75.638%                 | 0.866                            | ~2s  |
| CUR                   | 2.292e-14 | 84.361%                 | 0.954                            | ~1s  |
| CUR with 90% variance | 0.901     | 71.170%                 | 0.744                            | ~1s  |
