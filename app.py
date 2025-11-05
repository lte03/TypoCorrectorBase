from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import jpype
from typing import Dict, Optional
import gc
import time

from model import get_model, load_model_weights, correct_word
from n_gram_corrector import correct_sentence, load_ngram_frequencies

TORCH_SC_MODEL = None
UNIGRAM_PATH = "./lib/unigrams.txt"
BIGRAM_PATH = "./lib/bigrams.txt"
TRIGRAM_PATH = "./lib/trigrams.txt"
PROCESSING_TIMEOUT = 5

class BaseSpellCorrector:
    def check(self, word: str) -> bool:
        raise NotImplementedError

    def correct(self, word: str) -> str:
        raise NotImplementedError

    def cleanup(self):
        pass

class ZemberekSpellCorrector(BaseSpellCorrector):
    def __init__(self):
        self.morphology = TurkishMorphology.createWithDefaults()
        self.spell_checker = TurkishSpellChecker(self.morphology)

    def check(self, word: str) -> bool:
        return bool(self.spell_checker.check(word))

    def correct(self, word: str) -> str:
        suggestions = self.spell_checker.suggestForWord(word)
        return suggestions[0] if suggestions else "No suggestion"

    def cleanup(self):
        self.morphology = None
        self.spell_checker = None
        gc.collect()

class TorchSpellCorrector(BaseSpellCorrector):
    def __init__(self):
        global TORCH_SC_MODEL
        if TORCH_SC_MODEL is None:
            TORCH_SC_MODEL = get_model()
            load_model_weights(TORCH_SC_MODEL, "./lib/model_final.pth")
        self.model = TORCH_SC_MODEL
        self.morphology = TurkishMorphology.createWithDefaults()
        self.spell_checker = TurkishSpellChecker(self.morphology)

    def check(self, word: str) -> bool:
        return bool(self.spell_checker.check(word))

    def correct(self, word: str) -> str:
        return correct_word(self.model, word)

    def cleanup(self):
        self.morphology = None
        self.spell_checker = None
        gc.collect()

class LevensteinSpellCorrector(BaseSpellCorrector):
    def __init__(self):
        self.unigram = load_ngram_frequencies(UNIGRAM_PATH)
        self.bigram = load_ngram_frequencies(BIGRAM_PATH)
        self.trigram = load_ngram_frequencies(TRIGRAM_PATH)
        self.morphology = TurkishMorphology.createWithDefaults()
        self.spell_checker = TurkishSpellChecker(self.morphology)

    def check(self, word: str) -> bool:
        return bool(self.spell_checker.check(word))

    def correct(self, word: str) -> str:
        return correct_sentence(word, self.unigram, self.bigram, self.trigram)

    def process_text(self, text: str) -> Dict:
        start_time = time.time()
        words = text.strip().split()
        corrections = {}
        misspelled = []

        for word in words:
            if time.time() - start_time > PROCESSING_TIMEOUT:
                return {
                    "timeout": True,
                    "processed_words": corrections,
                    "misspelled": misspelled
                }
            
            if not self.check(word):
                corrected = correct_sentence(word, self.unigram, self.bigram, self.trigram)
                if corrected != word:
                    corrections[word] = corrected
                    misspelled.append(word)

        return {
            "timeout": False,
            "processed_words": corrections,
            "misspelled": misspelled
        }

    def cleanup(self):
        self.unigram = None
        self.bigram = None
        self.trigram = None
        self.morphology = None
        self.spell_checker = None
        gc.collect()

class ModelConfig:
    def __init__(self, id: str, name: str, description: str):
        self.id = id
        self.name = name
        self.description = description

class ModelManager:
    def __init__(self):
        self.current_model: Optional[BaseSpellCorrector] = None
        self.current_model_name: Optional[str] = None
        self.configurations = [
            ModelConfig("model1", "Zemberek Spell Checker", 
                      "Best for basic spell checking and simple corrections"),
            ModelConfig("model2", "Torch word-based Spell Checker", 
                      "Includes advanced grammar rules and style suggestions"),
            ModelConfig("model3", "N-Gram", 
                      "Uses context awareness for more accurate corrections")
        ]

    def get_model(self, model_name: str) -> BaseSpellCorrector:
        if self.current_model_name != model_name:
            if self.current_model:
                self.current_model.cleanup()
                self.current_model = None
        
            if model_name == "model1":
                self.current_model = ZemberekSpellCorrector()
            elif model_name == "model2":
                self.current_model = TorchSpellCorrector()
            elif model_name == "model3":
                self.current_model = LevensteinSpellCorrector()
        
            self.current_model_name = model_name
            gc.collect()
        
        return self.current_model

    def get_model_configs(self):
        return self.configurations

app = Flask(__name__)
CORS(app)

model_manager = ModelManager()
misspelled_words = []

@app.route('/')
def hello_world():
    model_configs = model_manager.get_model_configs()
    return render_template('index.html', 
                         title="Yazım Düzeltici",
                         misspelled_words=misspelled_words,
                         models=model_configs)

@app.route("/check", methods=["POST"])
def checker():
    data = request.get_json()
    word = data.get("word", "").strip()
    model_name = data.get("model", "model1")
    
    if not word:
        return jsonify({"error": "No word provided"}), 400
    
    model = model_manager.get_model(model_name)
    state = model.check(word)
    if not state:
        misspelled_words.append(word)
    
    return jsonify({
        "spelling": str(state)
    })

@app.route("/correct", methods=["POST"])
def corrector():
    data = request.get_json()
    word = data.get("word", "").strip()
    model_name = data.get("model", "model1")

    if not word:
        return jsonify({"error": "No word provided"}), 400

    model = model_manager.get_model(model_name)
    corrected = model.correct(word)
    
    return jsonify({
        "corrected": str(corrected)
    })

@app.route("/update_word", methods=["POST"])
def update_word():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        old_word = data.get("old_word", "").strip()
        new_word = data.get("new_word", "").strip()
        
        if not old_word or not new_word:
            return jsonify({"error": "Invalid data: old_word and new_word are required"}), 400
            
        global misspelled_words
        
        if old_word in misspelled_words:
            index = misspelled_words.index(old_word)
            misspelled_words[index] = new_word
            updated = True
        else:
            updated = False
        
        return jsonify({
            "success": True,
            "message": "Word updated successfully",
            "misspelled_words": misspelled_words,
            "updated": updated
        })
        
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500
    
@app.route("/process_text", methods=["POST"])
def process_text():
    data = request.get_json()
    text = data.get("text", "").strip()
    model_name = data.get("model", "model3")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    model = model_manager.get_model(model_name)
    
    if model_name == "model3":
        result = model.process_text(text)
        global misspelled_words

        corrected_text = text
        for word, corrected_word in result["processed_words"].items():
            escaped_word = word.replace(r'([.*+?^${}()|\[\]\\])', r'\\\1')
            corrected_text = corrected_text.replace(word, corrected_word)

        misspelled_words.extend(result["misspelled"])
        return jsonify({
            "corrected_text": corrected_text,
            "processed_words": result["processed_words"],
            "misspelled": result["misspelled"]
        })
    else:
        return jsonify({"error": "Text processing only available for Levenstein model"}), 400

if __name__ == '__main__':
    jvm_path = jpype.getDefaultJVMPath()
    zemberek_jar = "./lib/zemberek-full.jar"
    
    if not jpype.isJVMStarted():
        jpype.startJVM(jvm_path, f"-Djava.class.path={zemberek_jar}")
        
    TurkishTokenizer = jpype.JClass("zemberek.tokenization.TurkishTokenizer")
    TurkishMorphology = jpype.JClass("zemberek.morphology.TurkishMorphology")
    TurkishSpellChecker = jpype.JClass("zemberek.normalization.TurkishSpellChecker")
    
    app.run(debug=True)
