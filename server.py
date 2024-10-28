from flask import Flask, request, jsonify
from flask_cors import CORS
from vncorenlp import VnCoreNLP
import os

# Tạo ứng dụng Flask
app = Flask(__name__)
CORS(app)

# Khởi tạo VnCoreNLP với mô hình (thay đổi đường dẫn cho đúng)
current_dir = os.path.dirname(os.path.abspath(__file__))
jar_path = os.path.join(current_dir, "VnCoreNLP-1.2.jar")
vncorenlp = VnCoreNLP(jar_path, annotators="wseg,pos", max_heap_size='-Xmx500m')

@app.route('/analyze', methods=['POST'])
def analyze_sentence():
    data = request.json
    sentence = data.get('sentence')

    if not sentence:
        return jsonify({'error': 'Câu không được để trống'}), 400

    try:
        # Phân tích câu
        result = vncorenlp.annotate(sentence)
        tokens = result['sentences'][0]

        # Trích xuất thông tin từ phân tích POS
        analysis = [{'word': token['form'], 'pos': token['posTag']} for token in tokens]

        return jsonify({'analysis': analysis})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
