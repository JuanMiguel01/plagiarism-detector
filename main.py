from flask import Flask
from gui.app import app as gui_app
from back.segmentation import PlagiarismDetector
from back.preproccesing import TextPreprocessor
from back.lexical_similarity import calcular_similitud_coseno
from back.semantic_similarity import calcular_similitud_semantica
from back.quote_extraction import citation_check
from back.reading_tools import read_file
from back.lexical_similarity import show

# Inicializa los módulos necesarios
preprocessor = TextPreprocessor()
detector = PlagiarismDetector(preprocessor)

# Crea una instancia de la aplicación Flask pasando los módulos necesarios
app = Flask(__name__, template_folder='gui/templates')
app.register_blueprint(gui_app)

# Pasa los módulos a app.py
gui_app.preprocessor = preprocessor
gui_app.detector = detector
gui_app.calcular_similitud_coseno = calcular_similitud_coseno
gui_app.calcular_similitud_semantica = calcular_similitud_semantica
gui_app.citation_check = citation_check
gui_app.read_file = read_file
gui_app.show = show

if __name__ == '__main__':
    app.run(debug=True)
