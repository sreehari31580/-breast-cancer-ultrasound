import cv2
import numpy as np

def build_gabor_kernels(ksize=31, sigmas=[4.0], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4], lambdas=[10.0], gammas=[0.5]):
    kernels = []
    for theta in thetas:
        for sigma in sigmas:
            for lambd in lambdas:
                for gamma in gammas:
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
    return kernels

def extract_gabor_features(img, kernels):
    feats = []
    for kernel in kernels:
        filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        feats.append(filtered.mean())
        feats.append(filtered.std())
    return np.array(feats)