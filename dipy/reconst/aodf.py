import numpy as np
from scipy.stats import multivariate_normal



def gaussian_cone(h, r):
    mu = np.asarray([0, 0])
    sigma = np.asarray([[0.5, 0], [0, 0.5]])
    fall = np.zeros((h * r + 1, h * r + 1, h));
    uu=h
    for u in range(0, h):
        rr = 2
        if (u == h):
            utmp = h - 1
            s = ((h - utmp) / h) * rr * h / 2
        else:
            s = ((h - u) / h) * rr * h / 2

        if s > 0:
            x1 = np.arange(-1, 1, 1 / s)
            x2 = np.arange(-1, 1, 1 / s)

        X1, X2 = np.meshgrid(x1, x2)
        X1 = np.reshape(X1, (-1, 1))
        X2 = np.reshape(X2, (-1, 1))
        F = multivariate_normal.pdf(np.concatenate((X1, X2), axis=1), mu, sigma)
        print(F.shape)
        F = F.reshape((x1.shape[0]), x2.shape[0])
        F = (F - np.min(F)) / (np.max(F) - np.min(F))
        pd = (fall.shape[0] - fall.shape[0]) // 2 #??????????

        if ((h - u) == 0 | (h - u) == 1):
            fall[pd: h * r - pd, pd: h * r - pd, u] = F
        else:
            fall[1 + pd: h * r + 1 - pd, 1 + pd: h * r + 1 - pd, u + 1]

        uu = uu - 1

        fall[np.int32(np.round(h * r / 2)), np.int32(np.round(h * r / 2)), np.int32(u)] = 1

    fall_r = fall[:, :,  ::-1];




h=2
r=2
FallR = gaussian_cone(h,r)