from gemma_model import get_finetuned_pipeline

pipe_finetuned = get_finetuned_pipeline()
messages = [
    {
        "role": "user",
        "content": "제주도 애월읍에 있는 파스타를 먹고 싶은데, 추천해줘"
    }
]
prompt = pipe_finetuned.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe_finetuned(
    prompt,
    do_sample=True,
    temperature=0.7,
    top_k=20,
    top_p=0.95,
    repetition_penalty=1.2,
    max_new_tokens=150,  # 최대 토큰 개수 조정
    add_special_tokens=True
)
print(outputs[0]["generated_text"][len(prompt):])
