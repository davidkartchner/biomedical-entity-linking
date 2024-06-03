import subprocess

class Model_Wrapper:
    name: str  # name of the model
    model: object  # model object
    train_script_path: str  # path to the training script
    evaluate_script_path: str  # path to the evaluation script
    params: dict  # model arguments for training and inferencing

    @classmethod
    def load_arboel_biencoder(cls, name, params):
        model = model
        train_script_path = f"bioel/models/{name}/biencoder/train.py"
        evaluate_script_path = f"bioel/models/{name}/biencoder/evaluate.py"
        return cls(model, name, train_script_path, evaluate_script_path, params)

    @classmethod
    def load_arboel_crossencoder(cls, name, params):
        model = model
        train_script_path = f"bioel/models/{name}/crossencoder/train.py"
        evaluate_script_path = f"bioel/models/{name}/crossencoder/evaluate.py"
        return cls(model, name, train_script_path, evaluate_script_path, params)

    @classmethod
    def load_sapbert(cls, name, params):
        pass

    @classmethod
    def load_krissbert(cls, name, params):
        pass

    @classmethod
    def load_scispacy(cls, name, params):
        pass

    @classmethod
    def load_biobart(cls, name, params):
        pass

    @classmethod
    def load_biogenel(cls, name, params):
        pass

    def training(self):
        # Call the specified train.py script using subprocess, passing the dict as an argument
        subprocess.run(["python", self.train_script_path, "--params", self.params])

    def inference(self):
        # Call the specified evalaute.py script using subprocess, passing the dict as an argument
        subprocess.run(["python", self.evaluate_script_path, "--params", self.params])
