from transformers import AutoConfig, AutoTokenizer
import onnxruntime
import numpy as np

# 1. Load config, processor, and model
path_to_model = "../models/gemma-3-1b-it-ONNX"
config = AutoConfig.from_pretrained(path_to_model)
tokenizer = AutoTokenizer.from_pretrained(path_to_model)
decoder_session = onnxruntime.InferenceSession(f"{path_to_model}/onnx/model.onnx")

## Set config values
num_key_value_heads = config.num_key_value_heads
head_dim = config.head_dim
num_hidden_layers = config.num_hidden_layers
eos_token_id = 106 # 106 is for <end_of_turn>

# 2. Prepare inputs
## Create input messages
messages = [
  { "role": "system", "content": "You are a helpful assistant." },
  { "role": "user", "content": "Write me a poem about Machine Learning." },
]

## Apply tokenizer
template_output = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(template_output)

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np")

## Prepare decoder inputs
batch_size = inputs['input_ids'].shape[0]
past_key_values = {
    f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ('key', 'value')
}
input_ids = inputs['input_ids']
position_ids = np.tile(np.arange(1, input_ids.shape[-1] + 1), (batch_size, 1))

# 3. Generation loop
max_new_tokens = 1024
generated_tokens = np.array([[]], dtype=np.int64)
for i in range(max_new_tokens):
  logits, *present_key_values = decoder_session.run(None, dict(
      input_ids=input_ids,
      position_ids=position_ids,
      **past_key_values,
  ))

  ## Update values for next generation loop
  input_ids = logits[:, -1].argmax(-1, keepdims=True)
  position_ids = position_ids[:, -1:] + 1
  for j, key in enumerate(past_key_values):
    past_key_values[key] = present_key_values[j]

  generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
  if (input_ids == eos_token_id).all():
    break

  ## (Optional) Streaming
  print(tokenizer.decode(input_ids[0]), end='', flush=True)
print()

# 4. Output result
print(tokenizer.batch_decode(generated_tokens))
