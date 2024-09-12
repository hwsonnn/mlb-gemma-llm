from gemma_model import get_finetuned_tokenizer_model

# 저장된 모델과 토크나이저를 불러오기
model, tokenizer = get_finetuned_tokenizer_model()

# 7. 추론
def generate_recommendation(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(inputs['input_ids'], max_length=50, num_beams=5, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 사용자가 요청하는 예시: "제주도에서 한식집을 추천해줘."
recommendation = generate_recommendation("제주도에서 한식집을 추천해줘.")
print(recommendation)
