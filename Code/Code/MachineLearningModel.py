class MachineLearningModel:
    def load_model(self, filename):
        self.model = self.model.load(filename)

    def save_model(self, filename):
        self.model.save(filename)