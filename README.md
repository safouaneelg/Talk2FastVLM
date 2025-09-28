# Talk2FastVLM

A Flask web application for real-time interaction with the FastVLM-0.5B model using ONNX runtime on CPU. Capture live video from your webcam, provide voice or text prompts, and generate concise image descriptions or responses streamed in real-time. Supports continuous voice recognition, prompt locking for automated captioning, and aspect-ratio-preserving video rendering with black bars for unfilled areas.

Try my huggingface space for this app (with latency because of free tier usage) available at [https://huggingface.co/spaces/safouaneelg/Talk2FastVLM](https://huggingface.co/spaces/safouaneelg/Talk2FastVLM).

## Demo

[DEMO.webm](https://github.com/user-attachments/assets/679db7b4-67b0-45f2-9b64-23782b5e0fec)

## Installation

Follow these steps to set up and run the app locally.

### 1. Create a Virtual Environment
To avoid conflicts, use `conda` or Python `venv` to create an isolated environment:

**Using Conda:**
```bash
conda create -n talk2fastvlm python=3.10
conda activate talk2fastvlm
```

**Using venv:**
```bash
python -m venv talk2fastvlm
source talk2fastvlm/bin/activate  # On Windows: talk2fastvlm\Scripts\activate
```

### 2. Clone the Repository
```bash
git clone https://github.com/safouaneelg/Talk2FastVLM.git
cd Talk2FastVLM/
```

### 3. Download Model Files with Git LFS
The model files are hosted on Hugging Face and require Git LFS for large files:
```bash
git lfs install  # Install LFS if not already (one-time setup)
git lfs clone https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX
```
This downloads the ONNX model files to a `./FastVLM-0.5B-ONNX/onnx/` directory along with the model tokenizer, configs ...etc.

You can also manually download 3 quantized onnx files (vision encoder + decoder + embed) and store them in `FastVLM-0.5B-ONNX/onnx/`

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
Start the Flask server:
```bash
python app.py
```
The app will be available at `http://localhost:7860`. Open this URL in a modern browser (Chrome/Firefox recommended for speech recognition).

- **Access Permissions**: Grant camera and microphone access when prompted.
- **Voice Mode**: Enabled by default; click the mic icon to toggle off/on.
- **Prompt Locking**: Use the lock button to fix a prompt and enable auto-captioning every 5 seconds.

## Model Configuration
The app uses the `q4f16` quantized version by default for a balance of speed and quality (282 MB decoder, 272 MB embed, 253 MB vision). To switch quantizations:

1. Edit `app.py` and update the file names in `load_model()`:
   ```python
   vision_session = ort.InferenceSession(os.path.join(onnx_path, "vision_encoder_<variant>.onnx"), providers=providers)
   embed_session = ort.InferenceSession(os.path.join(onnx_path, "embed_tokens_<variant>.onnx"), providers=providers)
   decoder_session = ort.InferenceSession(os.path.join(onnx_path, "decoder_model_merged_<variant>.onnx"), providers=providers)
   ```
   Replace `<variant>` with one of the available options below.

2. Restart the app (`python app.py`).

### Available Quantization Variants
Thanks to [onnx-community](https://huggingface.co/onnx-community), these are the ONNX model files available from the Hugging Face repo.

| Variant       | Decoder Size | Embed Size | Vision Size | 
|---------------|--------------|------------|-------------|
| (default) `q4f16` | 282 MB | 272 MB | 253 MB | 
| `bnb4`        | 287 MB | 544 MB | 505 MB | 
| `fp16`        | 992 MB | 272 MB | 253 MB | 
| `int8`        | 503 MB | 136 MB | 223 MB | 
| `q4`          | 317 MB | 544 MB | 505 MB | 
| `quantized`   | 503 MB | 136 MB | 223 MB | 
| `uint8`       | 503 MB | 136 MB | 223 MB | 

- **Recommendations**:
  - `q4f16` for most users (fast on CPU).
  - `fp16` for better accuracy.
  - Lower-bit variants like `int8` or `uint8` for memory-constrained setups.

## Usage
1. Open `http://localhost:7860` in your browser.
2. Allow camera/mic access.
3. **Voice Input**: Speak your prompt (example "Describe my gesture/"); it auto-fills the text area and triggers generation.
4. **Text Input**: Type a prompt (e.g., default: "Describe this image in detail, focusing on any visible hands or gestures") and press Enter or click Send.
5. **Lock Mode**: Click the lock icon to fix the typed prompt and enable continuous captioning (updates every 5s).
6. Captions stream in real-time below the video.

The system prompt encourages concise, one-sentence responses focused on visible actions (e.g., hands/gestures).

## Troubleshooting
- **Speech Recognition Issues**: Ensure HTTPS or localhost; check browser console for errors. Not supported in all browsers.
- **Model Loading Errors**: Verify Git LFS download completed all files (~8 GB total). Or download manually the onnx files needed and store them in `./FastVLM-0.5B-ONNX/onnx` folder.
- **Port Conflicts**: The port `7860` was selected for huggingface but you can change it in `app.py` if needed.

## Licenses
[apple-amlr](LICENCE)
[model license](LICENSE_MODEL)

## Acknowledgments
- Built on [FastVLM](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX).
- Fast VLM original repo : [apple/ml-fastvlm](https://github.com/apple/ml-fastvlm)
