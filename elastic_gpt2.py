from flask import Flask, request, jsonify
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#flask app
app = Flask(__name__, static_url_path='', static_folder='project')

#set the model directory
model_directory = "model"

# Load the fine-tuned model and tokenizer from the directory
model = GPT2LMHeadModel.from_pretrained(model_directory)
tokenizer = GPT2Tokenizer.from_pretrained(model_directory)

# remove echoing the question
def remove_echo(response, user_input):
    response_tokens = tokenizer.tokenize(response)
    input_tokens = tokenizer.tokenize(user_input)
    
    # Remove echoed tokens at the beginning
    while len(response_tokens) > len(input_tokens) and response_tokens[:len(input_tokens)] == input_tokens:
        response_tokens = response_tokens[len(input_tokens):]
    
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(response_tokens), skip_special_tokens=True)

@app.route('/ask', methods=['POST'])
def ask():
    # Check if the request contains JSON data
    if not request.is_json:
        return jsonify({'error': 'Expected JSON data'}), 400
    data = request.json
    # Check if 'question' key is present in the request data (might be redundant)
    if 'question' not in data:
        return jsonify({'error': 'Key "question" missing in request data'}), 400
    question = data['question']

    # Use the GPT-2 model to generate an answer
    input_ids = tokenizer.encode(question, return_tensors='pt')
    attention_mask = torch.tensor([1] * len(input_ids[0]), dtype=torch.long).unsqueeze(0)
    max_output_length = 100  # Desired output length
    max_length = len(input_ids[0]) + max_output_length
    output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    max_length=max_length,
    num_return_sequences=1,
    do_sample=False,        # Beam search is deterministic
    num_beams=10,            # Setting beam width to 10
    no_repeat_ngram_size=2, # Prevent phrase repetitions
    length_penalty=0.3,     # try to keep it brief
    early_stopping=True     # Stop when the model outputs an EOS token
)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # assign the value of cleaned response
    cleaned_answer = remove_echo(answer, question)
    # return response minus echoed question
    return jsonify({'answer': cleaned_answer})

# flask app
@app.route('/')
def index():
    return app.send_static_file('chat.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)