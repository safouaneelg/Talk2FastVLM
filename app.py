from flask import Flask, render_template, request, Response, jsonify
from transformers import AutoTokenizer, AutoImageProcessor, TextIteratorStreamer
from PIL import Image
import torch
import base64
import io
import onnxruntime as ort
import numpy as np
import os
import queue  # For simple serialization
import threading

app = Flask(__name__)

model_id = "./FastVLM-0.5B-ONNX"  # Local path
IMAGE_TOKEN_INDEX = -200
tokenizer = None
image_processor = None
vision_session = None
embed_session = None
decoder_session = None
providers = ['CPUExecutionProvider']  # Add 'CUDAExecutionProvider' if using GPU
past_input_names = None
past_shapes = None
inference_queue = queue.Queue()  # Serialize inferences (1 at a time)
queue_lock = threading.Lock()

def load_model():
    global tokenizer, image_processor, vision_session, embed_session, decoder_session, past_input_names, past_shapes
    if tokenizer is None:
        print("Loading tokenizer and image processor...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        print("Tokenizer and processor loaded!")

    if vision_session is None:
        print("Loading ONNX sessions...")
        onnx_path = os.path.join(model_id, "onnx")
        vision_session = ort.InferenceSession(os.path.join(onnx_path, "vision_encoder_q4f16.onnx"), providers=providers)
        embed_session = ort.InferenceSession(os.path.join(onnx_path, "embed_tokens_q4f16.onnx"), providers=providers)
        decoder_session = ort.InferenceSession(os.path.join(onnx_path, "decoder_model_merged_q4f16.onnx"), providers=providers)
        
        # Inspect inputs/outputs (unchanged)
        print("Vision inputs:", [inp.name for inp in vision_session.get_inputs()])
        print("Vision outputs:", [out.name for out in vision_session.get_outputs()])
        print("Embed inputs:", [inp.name for inp in embed_session.get_inputs()])
        print("Embed outputs:", [out.name for out in embed_session.get_outputs()])
        print("Decoder inputs:", [inp.name for inp in decoder_session.get_inputs()])
        print("Decoder outputs:", [out.name for out in decoder_session.get_outputs()])
        
        # Collect past_key_values info for KV cache
        global past_input_names, past_shapes
        past_input_names = [inp.name for inp in decoder_session.get_inputs() if 'past_key_values' in inp.name]
        past_shapes = [inp.shape for inp in decoder_session.get_inputs() if 'past_key_values' in inp.name]
        print("Past input names:", past_input_names)
        print("Past shapes:", past_shapes)  # Run once to verify; e.g., [1, 16, -1, 64] for keys/values
        
        print("ONNX sessions loaded!")

load_model()

def generate_tokens(image, user_prompt):
    # Same prompt preparation (unchanged)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. If you see a person in the image, this is me interacting, so if the question involve me as a personal, answer it based on my actions. Always answer very briefly and in one sentence while being concise, using only essential information."},
        {"role": "user", "content": f"<image>\n{user_prompt}"}
    ]
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    pre, post = rendered.split("<image>", 1)
    
    pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
    
    # Process image (client already resized; processor will handle final)
    px = image_processor(images=image, return_tensors="pt")["pixel_values"]
    
    # Run vision encoder
    vision_feat_np = vision_session.run(None, {"pixel_values": px.numpy()})[0]
    vision_features = torch.from_numpy(vision_feat_np)
    
    # Embed text parts separately (avoid dummy <image> token)
    pre_emb_np = embed_session.run(None, {"input_ids": pre_ids.numpy()})[0]
    post_emb_np = embed_session.run(None, {"input_ids": post_ids.numpy()})[0]
    pre_embeds = torch.from_numpy(pre_emb_np)
    post_embeds = torch.from_numpy(post_emb_np)
    
    # Insert vision features
    input_embeds = torch.cat([pre_embeds, vision_features, post_embeds], dim=1)
    num_vision_tokens = vision_features.shape[1]
    seq_len = pre_ids.shape[1] + num_vision_tokens + post_ids.shape[1]
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)
    position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
    
    # Initial KV cache: zeros with concrete shapes (from past_shapes pattern: ['batch_size', 2, 'past_sequence_length', 64])
    initial_past_feed = {}
    concrete_shape = [1, 2, 0, 64]  # batch=1, heads=2, seq=0, head_dim=64
    for name in past_input_names:
        initial_past_feed[name] = np.zeros(concrete_shape, dtype=np.float16)
    
    # First decoder run (on full prompt)
    decoder_inputs = {
        "inputs_embeds": input_embeds.numpy(),
        "attention_mask": attention_mask.numpy(),
        "position_ids": position_ids.numpy(),
    }
    decoder_inputs.update(initial_past_feed)
    
    decoder_outputs = decoder_session.run(None, decoder_inputs)
    logits = torch.from_numpy(decoder_outputs[0])
    
    eos_token_id = tokenizer.eos_token_id
    max_new_tokens = 128
    past_feed = dict(zip(past_input_names, decoder_outputs[1:]))  # Update KV from presents
    current_pos = seq_len
    
    for step in range(max_new_tokens):
        # Sample next token (greedy)
        next_token_ids = torch.argmax(logits[:, -1, :], dim=-1)  # (1,)
        next_token_id = next_token_ids[0].item()
        
        if next_token_id == eos_token_id:
            break
        
        # Yield decoded token
        new_text = tokenizer.decode(next_token_ids, skip_special_tokens=True)
        yield new_text
        
        # Embed next token
        next_input_ids = next_token_ids.unsqueeze(1)  # (1, 1)
        next_emb_np = embed_session.run(None, {"input_ids": next_input_ids.numpy()})[0]
        input_embeds = torch.from_numpy(next_emb_np)  # (1, 1, hidden)
        
        # Update position_ids and attention_mask for single token
        position_ids = torch.tensor([[current_pos]], dtype=torch.long)
        attention_mask = torch.cat([attention_mask, torch.tensor([[1]], dtype=torch.long)], dim=1)
        
        # Next decoder run (incremental)
        decoder_inputs = {
            "inputs_embeds": input_embeds.numpy(),
            "attention_mask": attention_mask.numpy(),
            "position_ids": position_ids.numpy(),
        }
        decoder_inputs.update(past_feed)
        
        decoder_outputs = decoder_session.run(None, decoder_inputs)
        logits = torch.from_numpy(decoder_outputs[0])
        
        # Update KV cache
        past_feed = dict(zip(past_input_names, decoder_outputs[1:]))
        current_pos += 1

# Queue wrapper for serialized inference
def queued_generate(image, user_prompt):
    with queue_lock:
        if not inference_queue.empty():
            # Wait for previous to finish (simple poll; upgrade to asyncio if needed)
            while not inference_queue.empty():
                pass
        inference_queue.put(1)
    try:
        for token in generate_tokens(image, user_prompt):
            yield token
    finally:
        inference_queue.get()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Handle FormData
        image_file = request.files.get('image')
        user_prompt = request.form.get('prompt', '')
        
        if not image_file:
            return jsonify({'error': 'No image provided'}), 400
        
        image_data = image_file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        def stream():
            for token in queued_generate(image, user_prompt):
                yield token
        
        return Response(stream(), mimetype='text/plain')
    except Exception as e:
        print(f"Error in generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=7860)
