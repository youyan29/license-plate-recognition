import cv2
from MachineLearningModel import MachineLearningModel


class SVMModel(MachineLearningModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train_model(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict_chars(self, samples):
        result = self.model.predict(samples)
        return result[1].ravel()