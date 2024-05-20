import numpy as np
from scipy.spatial.distance import cdist

# Kernel classes from kernel_fn.py
class AbsKernel:  # Abstract class to provide NotImplementedError for children
    def fit(self, data: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cal_kernel_mat_grad(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        # Compute gradient of kernel matrix with respect to data2.
        raise NotImplementedError

    def cal_kernel_mat_jacob(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        # Compute jacobi matrix of kernel matrix. dx1 dx2 k(x1, x2)
        # assert data1 and data2 are single dimensions
        raise NotImplementedError

class BinaryKernel(AbsKernel):
    def __init__(self):
        super(BinaryKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs):
        pass

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        res = data1.dot(data2.T)
        res += (1 - data1).dot(1 - data2.T)
        return res

class GaussianKernel(AbsKernel):
    sigma: np.float64

    def __init__(self):
        super(GaussianKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.sigma = np.median(dists)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        return np.exp(-dists)

#* Closed-form leave-one-out cross-validation for lambda
def cal_loocv(K, y, lam):
    nData = K.shape[0]
    I = np.eye(nData)
    H = I - K.dot(np.linalg.inv(K + lam * nData * I))
    tildeH_inv = np.diag(1.0 / np.diag(H))
    return np.linalg.norm(tildeH_inv.dot(H.dot(y)))

#* Main class, modified from ATEBackDoorMeanEmbeddingCI
class MeanEmbeddingModel:
    #// Note that, for the Gaussian kernel, the resulting kernel matrix can be interpreted as an elementwise product of variable-specific matrices.
    #// This is not necessarily true for other kernels, e.g., the linear kernel.
    treatment_kernel_fn = BinaryKernel()  # Treatment is binary in the data used
    covariates_kernel_fn = GaussianKernel()
    #// Parentheses important here so that these are treated as bound methods rather than as functions (in Python 3; unbound methods in Python 2).
    #// For other kernel choices, see kernel_fn.py.

    def __init__(self, treatment, covariates, lam, nu, **kwargs):
        self.lam = lam
        self.nu = nu
        self.treatment_kernel_fn.fit(treatment)
        self.covariates_kernel_fn.fit(covariates)

    def cal_lambda(self, outcome: np.ndarray, treatment: np.ndarray, covariates: np.ndarray):
        treatment_kernel = self.treatment_kernel_fn.cal_kernel_mat(treatment, treatment)
        covariates_kernel = self.covariates_kernel_fn.cal_kernel_mat(covariates, covariates)

        # n_data = treatment_kernel.shape[0]
        lam_score = [cal_loocv(treatment_kernel * covariates_kernel, outcome, lam) for lam in self.lam]
        self.lam = self.lam[np.argmin(lam_score)]

    def fit_alpha(self, treatment_nl: np.ndarray, covariates_nl: np.ndarray):
        self.treatment_nl = treatment_nl
        self.covariates_nl = covariates_nl
        self.n = self.treatment_nl.shape[0]
        self.ones = np.ones((self.n, 1))
        self.zeros = np.zeros((self.n, 1))
        
        covariates_kernel_nl = self.covariates_kernel_fn.cal_kernel_mat(self.covariates_nl, self.covariates_nl)
        K1 = self.treatment_kernel_fn.cal_kernel_mat(self.treatment_nl, self.treatment_nl) * covariates_kernel_nl
        K2 = (self.treatment_kernel_fn.cal_kernel_mat(self.ones, self.treatment_nl)
              - self.treatment_kernel_fn.cal_kernel_mat(self.zeros, self.treatment_nl)).T * covariates_kernel_nl
        K3 = K2.T
        K4 = (self.treatment_kernel_fn.cal_kernel_mat(self.ones, self.ones)
              - self.treatment_kernel_fn.cal_kernel_mat(self.ones, self.zeros)
              - self.treatment_kernel_fn.cal_kernel_mat(self.zeros, self.ones)
              + self.treatment_kernel_fn.cal_kernel_mat(self.zeros, self.zeros)) * covariates_kernel_nl
        
        Omega = np.block([[K1 @ K1, K1 @ K2],
                          [K3 @ K1, K3 @ K2]])
        K = np.block([[K1, K2],
                      [K3, K4]])
        v = np.sum(np.block([[K2],
                             [K4]]), axis = 1, keepdims = True)
        self.w = np.linalg.solve(Omega + self.n * self.lam * (K + self.nu * np.eye(2 * self.n)), v)
        
    def predict(self, treatment_l: np.ndarray, covariates_l: np.ndarray) -> np.ndarray:
        covariates_kernel_cross = self.covariates_kernel_fn.cal_kernel_mat(self.covariates_nl, covariates_l)
        
        u_top = self.treatment_kernel_fn.cal_kernel_mat(self.treatment_nl, treatment_l) * covariates_kernel_cross
        u_bottom = (self.treatment_kernel_fn.cal_kernel_mat(self.ones, treatment_l)
                    - self.treatment_kernel_fn.cal_kernel_mat(self.zeros, treatment_l)) * covariates_kernel_cross
        u = np.block([[u_top],
                      [u_bottom]])
        return (self.w.T @ u).T