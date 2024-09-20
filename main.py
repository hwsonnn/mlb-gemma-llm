from gemma_model import get_finetuned_pipeline

# # 저장된 모델과 토크나이저를 불러오기
# model, tokenizer = get_finetuned_tokenizer_model()

# # 7. 추론
# def generate_recommendation(input_text):
#     inputs = tokenizer(input_text, return_tensors="pt")
#     output = model.generate(inputs['input_ids'], max_length=50, num_beams=5, no_repeat_ngram_size=2)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # 사용자가 요청하는 예시: "제주도에서 한식집을 추천해줘."
# recommendation = generate_recommendation("제주도에서 한식집을 추천해줘.")
# print(recommendation)

pipe_finetuned = get_finetuned_pipeline()
doc = '제주도 애월 쪽 한식집을 추천해줘.'
messages = [
    {
        "role": "user",
        "content": "다음 글을 요약해주세요:\n\n{}".format(doc)
    }
]
prompt = pipe_finetuned.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe_finetuned(
    prompt,
    do_sample=True,
    temperature=0.2,
    top_k=50,
    top_p=0.95,
    add_special_tokens=True
)
print(outputs[0]["generated_text"][len(prompt):])
