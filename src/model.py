from transformers import AutoTokenizer, AutoModelForCausalLM

def get_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
    return tokenizer, model
