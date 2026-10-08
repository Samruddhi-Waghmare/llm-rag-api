from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

messages = [{"role": "user", "content": "What is backpropagation?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)

inputs = tokenizer(text, return_tensors="pt")
print(inputs["input_ids"])

output = model.generate(**inputs, max_new_tokens=150)

prompt_length = inputs["input_ids"].shape[1]
reply =  tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
print(reply)