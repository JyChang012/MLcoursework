from task1 import SVM_Functions as svm


def main():
    x, y = svm.loadData('./task2/task2.mat')
    svm.plotData(x, y, title='original data')
    K_matrix = svm.gaussianKernel(x, sigma=0.1)
    model = svm.svmTrain_SMO(x, y, C=0.1, kernelFunction='gaussian', K_matrix=K_matrix)
    svm.visualizeBoundaryGaussian(x, y, model, sigma=0.1)


if __name__ == '__main__':
    main()
