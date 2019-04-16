__author__ = 'Raja Teku'
import json
import sys
print(sys.path.append("."))
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
    path_to_config = "./model_config.json"
    in_sequence = sen.split(" ")
    model = load_model(path_to_config)
    tags = model.predict(in_sequence)
    sents= []
    current_sent = []
    text = sen.strip().split()
    for i, word in enumerate(text):
        if tags[i] == 'B-sent':
            if current_sent:
                sents.append(' '.join(current_sent))
            current_sent = [word]
        else:
            current_sent.append(word)

    sents.append(' '.join(current_sent))

    return sents

if __name__ == '__main__':
    pass
