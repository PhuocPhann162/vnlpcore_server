from flask import Flask, request, jsonify
from flask_cors import CORS
from vncorenlp import VnCoreNLP
import os
import spacy

# Tạo ứng dụng Flask
app = Flask(__name__)
CORS(app)

# Khởi tạo VnCoreNLP với mô hình 
current_dir = os.path.dirname(os.path.abspath(__file__))
jar_path = os.path.join(current_dir, "VnCoreNLP-1.2.jar")
vncorenlp = VnCoreNLP(jar_path, annotators="wseg,pos", max_heap_size='-Xmx500m')

# Khởi tạo spaCy cho tiếng Anh
nlp_en = spacy.load("en_core_web_sm")

@app.route('/analyze', methods=['POST'])
def analyze_sentence():
    data = request.json
    sentence = data.get('sentence')
    language = data.get('language', 'vi')

    if not sentence:
        return jsonify({'error': 'Câu không được để trống'}), 400

    try:
        if language == 'vi':  # Phân tích tiếng Việt
            result = vncorenlp.annotate(sentence)
            tokens = result['sentences'][0]
            analysis = [{'word': token['form'], 'pos': token['posTag']} for token in tokens]

        elif language == 'en':  # Phân tích tiếng Anh
            doc = nlp_en(sentence)
            analysis = [{'word': token.text, 'pos': token.pos_} for token in doc]

        else:
            return jsonify({'error': 'Ngôn ngữ không được hỗ trợ'}), 400

        return jsonify({'analysis': analysis})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
