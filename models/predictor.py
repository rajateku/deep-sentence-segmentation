__author__ = 'Raja Teku'
import json
from .ner_model import NERModel
from .config import Config

def load_model(config_path):
    config_params = json.load(open(config_path))
    config = Config(**config_params, load=True)
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)
    print("Completed loading the model")
    return model

def predict(sen):
    path_to_config = "/media/nava/sd2/raja-stuff/sen-tagging/sentence-tagging/model_config.json"
    in_sequence = sen.split(" ")
    model = load_model(path_to_config)
    pred_sequence = model.predict(in_sequence)
    return pred_sequence

if __name__ == '__main__':
    pass
