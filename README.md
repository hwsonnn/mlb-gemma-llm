# Gemma LLM Model Fine-Tuning

This repository contains the fine-tuned model gemma-2-2b-it-jeju-peft-restaurant-recommender-v1, specializing in recommending restaurants in Jeju Island. The model is based on the Gemma LLM and has been fine-tuned using techniques like LoRA, PEFT, and QLoRA for quantization.   
The training code can be found in the gemma-model/ directory under train.py at https://github.com/hwsonnn/mlb-gemma-llm. Please refer to it for details.

#### Table of Contents
- Introduction
- Dataset
- Project Structure
- Data Preprocessing
- Model Training
- Testing the Model
- Results
- Requirements
- License
- Introduction

#### Introduction
The goal of this project is to create a specialized language model that can provide restaurant recommendations in Jeju Island, South Korea. By fine-tuning the Gemma LLM model with domain-specific data, we enhance its ability to understand and respond to user queries related to local eateries.

#### Dataset
We used the "Tourism KVQA Data (Jeju Island and Island Regions)" from AI Hub, which contains detailed information about restaurants in Jeju Island. The dataset includes various attributes such as restaurant name, overview, address, menu, operating hours, and customer reviews.
```
"관광타입","음식점"
"음식점명","다래향"
"개요","조천읍 다래향은 제주 함덕해수욕장 인근에 있는 짬뽕 전문점이다. 신선한 해산물이 가득 들어간 다양한 종류의 짬뽕이 눈길을 사로잡는다. 짜장은 짬뽕 전문점답게 유니짜장 한 메뉴이다. 잡탕밥, 잡채밥 등의 메뉴도 제공한다. 테우, 오고생이가 인접해 있습니다."
"주소","제주특별자치도 제주시 조천읍 조함해안로 428-4"
"주소(도로명)","제주특별자치도 제주시 조천읍 조함해안로 428-4"
"주소(지번주소)","제주특별자치도 제주시 조천읍 함덕리 3132-1"
"대표번호","064-782-9466"
"영업시간","09:00 ~ 20:30"
"휴무일","매주 월요일"
"취급메뉴","차돌짬뽕, 다래향해물짬뽕"
"주차시설 유무","없음"
"리뷰키워드","음식이 맛있어요, 양이 많아요, 재료가 신선해요, 가성비가 좋아요, 친절해요"
```

### Project Structure
The project directory is organized as follows:
```
mlb-gemma-llm/
├── data/
│   └── 219_2_KVQA_jeju_tour_data/
├── gemma-model/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
├── gemma-2-2b-it-lora-fine-tuned/
├── lora_adapter
├── requirements.txt
└── README.md
```

#### Data Preprocessing
We preprocessed the data to create prompt-like inputs for the model. The data_loader.py script reads the CSV files and generates various question-answer pairs for each restaurant, formatted to resemble a conversation.

##### Data Loader Highlights
- **Reading CSV Files**: Parses restaurant information from CSV files in the dataset.
- **Generating Q&A Pairs**: Creates multiple question-answer pairs for each restaurant, including:
    - Restaurant recommendations based on location and menu.
    - Inquiries about operating hours, holidays, parking availability, and signature dishes.
- **Formatting Data**: Structures the data into a format suitable for model training using a conversational template.

#### Model Training
We utilized the **gemma-2-2b-it** model as the base and fine-tuned it using quantization techniques like **QLoRA** and **SFTTrainer**.

1. Clone the repository:  
    ```git clone https://github.com/hwsonnn/  mlb-gemma-llm``` 
2. Run the training script:  
    ```python gemma_model/train.py```   
3. Example:  
      

##### Training Highlights
- **LoRA Configuration**: Applied Low-Rank Adaptation (LoRA) to fine-tune the model efficiently by updating a subset of parameters.
- **Quantization**: Used 4-bit quantization to reduce memory usage and speed up training without significant loss in performance.
- **Training Script**: The train.py script handles model training, utilizing the SFTTrainer from the trl library for supervised fine-tuning.

#### Testing the Model
To test the performance of the fine-tuned model, we used a script that:
- Loads the fine-tuned model and tokenizer.
- Formats user queries into the appropriate prompt structure.
- Generates responses using the model.

##### Example Query
```
"제주도 애월읍에서 카레를 먹고 싶은데, 추천해줄 만한 곳 있어?"
```
##### Sample Output
```
"애월사이카레를 추천합니다! 주소는 제주 제주시 애월읍 애월해안로 752이고, 영업시간은 전10:00 - 19:00입니다. 주 메뉴는 흑돼지카레스튜, 카레짬뽕 이며, 안녕하세요. 랍스터빈, 파라펜션 이 인접해......"
```
*Note: The model provides recommendations based on the data it was trained on, delivering relevant information about restaurant options in the specified area.*

### Results
The fine-tuned model successfully provides accurate and contextually appropriate restaurant recommendations based on user queries. It effectively utilizes the specialized dataset to generate informative and helpful responses, demonstrating its capability as a conversational assistant for Jeju Island's dining options.

### Requirements
- Python 3.12 or higher
- PyTorch
- Transformers
- Datasets
- PEFT
- BitsAndBytes
- TRL
Install the required packages using:
```pip install -r requirements.txt```

### License
This project is licensed under the GEMMA License - see the [Hugging Face](https://huggingface.co/google/gemma-2-2b-it) page for details.

---
By leveraging the Gemma LLM model and fine-tuning it with domain-specific data, we've created a specialized assistant capable of providing restaurant recommendations in Jeju Island. The use of LoRA, PEFT, and QLoRA techniques allowed us to efficiently train the model while optimizing resource usage.

---
*For more details on the implementation and to explore the code, please refer to the individual scripts in the gemma-model/ directory.*