from flask_setup import app
from flask import jsonify, request
import logging
from models import predictor
ERROR = "error"
import nltk
import spacy
logger = logging.getLogger("app_logger")
from flask import render_template



nlp = spacy.load('en_core_web_sm')

@app.route("/")
def index():
    return render_template('index.html')

def remove_non_ascii(text):
    temp_list = []
    for i in text:
        if ord(i) < 128:
            temp_list.append(i)
        else:
            temp_list.append("")

    return "".join(temp_list)

@app.route("/", methods=['POST'])
def input_text():

    req = request.form['paragraph']
    print(req)

    req = remove_non_ascii(req)
    req3 = req.replace("&", " ")
    req4 = req3.replace("-", " ")
    sents = predictor.predict(req4)

    doc = nlp(req)
    spacy_sentences = []
    for sent in doc.sents:
        spacy_sentences.append(str(sent))

    obj = {
        "sentences": sents,
        "count": len(sents),
        "nltk_sentences": nltk.sent_tokenize(req),
        "nltk_count": len(nltk.sent_tokenize(req)),
        "spacy_sentences": spacy_sentences,
        "spacy_count": len(spacy_sentences)
    }

    return render_template('index.html', output=obj, paragraph = req , response = str(sents), response_nltk = str(nltk.sent_tokenize(req)), response_spacy = str(spacy_sentences))



@app.route("/healthcheck", methods=['GET'])
def healthcheck():
    return jsonify({"health": "ok"}), 200


@app.route("/sentence_detection", methods=['POST'])
def classify():

    if request.is_json:
        req = request.get_json()
    else:
        app.logger.info("Invalid json")
        return jsonify({ERROR: "INVALID_INPUT"}), 400

    sents = predictor.predict(req["para"])

    doc = nlp(req["para"])
    spacy_sentences = []
    for sent in doc.sents:
        spacy_sentences.append(str(sent))

    obj = {
        "sentences" : sents,
        "count" : len(sents),
        "nltk_sentences": nltk.sent_tokenize(req["para"]),
        "nltk_count": len(nltk.sent_tokenize(req["para"])),
        "spacy_sentences":spacy_sentences,
        "spacy_count":len(spacy_sentences)
    }

    return jsonify(obj), 200
