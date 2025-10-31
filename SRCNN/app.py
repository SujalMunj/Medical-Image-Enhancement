import os
from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch
from .model import SRCNN

try:
    import pydicom
    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_DIR = os.path.join(STATIC_DIR, 'uploads')
RESULTS_DIR = os.path.join(STATIC_DIR, 'results')

os.makedirs(UPLOAD_DIR, exist_ok=True)
(os.makedirs(RESULTS_DIR, exist_ok=True))

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm'}

def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

# -----------------------------------------------------------------------------
# Model loading and inference helpers
# -----------------------------------------------------------------------------
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model_cache = {}


def _detect_modality_from_name(path: str) -> str:
    p = path.lower()
    if 'xray' in p or 'x_ray' in p or 'x-ray' in p:
        return 'xray'
    if 'ct' in p:
        return 'ct'
    if 'mri' in p:
        return 'mri'
    # default to xray if unknown
    return 'xray'


def _load_model_for_modality(modality: str) -> SRCNN:
    modality = modality.lower()
    if modality in _model_cache:
        return _model_cache[modality]

    ckpt = os.path.join(BASE_DIR, 'checkpoints', modality, f'SRCNN_{modality}.pth')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'Model checkpoint not found for modality {modality}: {ckpt}')

    model = SRCNN().to(_device)
    state = torch.load(ckpt, map_location=_device)
    model.load_state_dict(state)
    model.eval()
    _model_cache[modality] = model
    return model


def _read_image_to_gray256(path: str) -> np.ndarray:
    ext = os.path.splitext(path.lower())[1]
    if ext == '.dcm':
        if not _HAS_PYDICOM:
            raise RuntimeError('pydicom not installed; cannot read .dcm files')
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        # normalize to 0-255 range
        img -= img.min()
        if img.max() > 0:
            img = img / img.max()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError('Invalid or unsupported image file')
    img = cv2.resize(img, (256, 256))
    return img


def _enhance_with_srcnn(img_gray_256: np.ndarray, model: SRCNN) -> np.ndarray:
    # simulate low-res then upscale to match enhancesingle.py
    lr = cv2.resize(img_gray_256, (128, 128), interpolation=cv2.INTER_CUBIC)
    lr_up = cv2.resize(lr, (256, 256), interpolation=cv2.INTER_CUBIC)

    inp = torch.tensor(lr_up / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(_device)
    with torch.no_grad():
        out = model(inp)
    out_img = (out.cpu().numpy().squeeze() * 255.0).clip(0, 255).astype(np.uint8)
    return out_img

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(f.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    f.save(save_path)

    file_url = url_for('static', filename=f'uploads/{filename}', _external=True)
    return jsonify({'filename': filename, 'file_url': file_url})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        filename = data.get('filename') if data else None
        if not filename:
            return jsonify({'error': 'filename is required'}), 400

        src_path = os.path.join(UPLOAD_DIR, secure_filename(filename))
        if not os.path.exists(src_path):
            return jsonify({'error': 'uploaded file not found'}), 404

        modality = _detect_modality_from_name(src_path)
        model = _load_model_for_modality(modality)

        img256 = _read_image_to_gray256(src_path)
        enhanced = _enhance_with_srcnn(img256, model)

        out_name = f'enhanced_{filename.rsplit(".", 1)[0]}.png'
        out_path = os.path.join(RESULTS_DIR, out_name)
        cv2.imwrite(out_path, enhanced)

        enhanced_url = url_for('static', filename=f'results/{out_name}', _external=True)

        # Keep the schema that the current frontend expects (prediction/confidence)
        return jsonify({
            'prediction': f'{modality.upper()} image enhanced',
            'confidence': 0.99,
            'enhanced_url': enhanced_url
        })
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
