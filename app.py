from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hqq.engine.hf import HQQModelForCausalLM
import threading

app = Flask(__name__)

# Configuration variables
device = 'cuda'
torch_dtype = torch.float16
load_full_model = False

# Load models and tokenizers
quantized_model_id = 'mobiuslabsgmbh/Llama-2-7b-chat-hf_1bitgs8_hqq'
full_model_id = 'meta-llama/Llama-2-7b-chat-hf'

one_bit_model = HQQModelForCausalLM.from_quantized(quantized_model_id, adapter='adapter_v0.1.lora')
one_bit_model.eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_id)

fullbit_model = None
if load_full_model:
    fullbit_model = AutoModelForCausalLM.from_pretrained(full_model_id, device_map=device, trust_remote_code=True, torch_dtype=torch_dtype).eval()

# Define the chat processing function
def chat_processor(chat, current_model, max_new_tokens=100, do_sample=True):
    # Prepare input and settings for generation
    input_ids = tokenizer.encode("<s> [INST] " + chat + " [/INST] ", return_tensors="pt").to(device)
    generate_params = {
        'input_ids': input_ids,
        'max_new_tokens': max_new_tokens,
        'do_sample': do_sample,
        'pad_token_id': tokenizer.pad_token_id,
        'top_p': 0.90 if do_sample else None,
        'top_k': 50 if do_sample else None,
        'temperature': 0.6 if do_sample else None,
        'num_beams': 1,
        'repetition_penalty': 1.2,
    }
    
    # Generate response
    outputs = current_model.generate(**generate_params)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    chat = data.get('chat')
    model_type = data.get('model_type', 'quantized')  # Default to quantized model

    if not chat:
        return jsonify({'error': 'Chat text is required'}), 400

    if model_type == 'quantized':
        model = one_bit_model
    elif model_type == 'full' and fullbit_model:
        model = fullbit_model
    else:
        return jsonify({'error': 'Invalid model type or full model not loaded'}), 400
    
    # Run the model in a separate thread to avoid blocking
    result = {}
    thread = threading.Thread(target=lambda: result.update({'response': chat_processor(chat, model)}))
    thread.start()
    thread.join()

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False,port=5001)

