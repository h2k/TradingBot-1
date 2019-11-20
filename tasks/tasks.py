import base64
from predication import Engine
from predication import SuggestionEngine


def predict_error(job, exc_type, exc_value, traceback):
    print(job, exc_type, exc_value, traceback)


def predict(stock, filename):
    engine = Engine(stock, filename)
    filepath = engine.train_model(save_results=True)
    with open(filepath, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def suggest(symbols):
    suggest_engine = SuggestionEngine(symbols)
    return suggest_engine.suggest()
