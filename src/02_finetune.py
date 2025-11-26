import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch


# ==========================================
# ⚙️ 설정 (내 환경에 맞게 수정)
# ==========================================
# 사용할 기본 모델 (small이 성능/속도 밸런스가 좋음)
MODEL_ID = "openai/whisper-small"
# 데이터 경로
DATA_ROOT = "../dataset"
CSV_FILE = os.path.join(DATA_ROOT, "metadata.csv")
AUDIO_FOLDER = os.path.join(DATA_ROOT, "raw_audio")
# 저장할 경로
OUTPUT_DIR = "../models/whisper-finetuned-v1"

# 학습 설정 (데이터가 적으므로 짧게 설정됨)
MAX_STEPS = 300          # 학습 반복 횟수 (데이터가 50개면 300~500 추천)
BATCH_SIZE = 4           # 한 번에 학습할 데이터 양 (GPU 메모리에 따라 조절)
LEARNING_RATE = 1e-3     # 학습률 (LoRA는 보통 1e-3 사용)

# ==========================================
# 1. 데이터셋 로드 및 전처리
# ==========================================
print(f"📂 데이터셋 로드 중... ({CSV_FILE})")

# CSV 파일이 없으면 에러
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError("metadata.csv 파일이 없습니다. 01_recorder.py로 데이터를 먼저 만드세요!")

# 데이터셋 생성
dataset = Dataset.from_csv(CSV_FILE)

# 오디오 경로 수정 (CSV에는 파일명만 있으므로 전체 경로로 변경)
def resolve_path(batch):
    batch["audio"] = os.path.join(AUDIO_FOLDER, batch["file_name"])
    return batch

dataset = dataset.map(resolve_path)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print("✅ 데이터셋 준비 완료!")

# ==========================================
# 2. 프로세서(Feature Extractor + Tokenizer) 준비
# ==========================================
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language="Korean", task="transcribe")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="Korean", task="transcribe")

def prepare_dataset(batch):
    # 오디오 로드 및 특성 추출
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # 텍스트를 라벨 ID로 변환
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("🔄 데이터 전처리 중...")
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# ==========================================
# 3. Data Collator (배치 처리를 위한 도구)
# ==========================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # padding 토큰(-100) 처리 (손실 계산 제외용)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # 시작 토큰이 있으면 잘라내기
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ==========================================
# 4. 모델 로드 및 LoRA 설정 (핵심!)
# ==========================================
print(f"🤖 모델 로드 중... ({MODEL_ID})")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# LoRA 설정 (모델 전체를 학습하지 않고 일부만 학습 -> 빠름)
config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none",
    # task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, config)
model.config.use_cache = False
model.print_trainable_parameters() # 학습 가능한 파라미터 수 출력



# ==========================================
# 5. 학습 시작
# ==========================================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=LEARNING_RATE,
    max_steps=MAX_STEPS,
    gradient_checkpointing=True,
    fp16=True, # GPU 지원 시 True, 아니면 False
    report_to="none",
    eval_strategy="no", # 시간 절약을 위해 평가 생략
    save_strategy="steps",
    save_steps=100,
    logging_steps=25,
    load_best_model_at_end=False,
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    data_collator=data_collator,
    # tokenizer=processor.feature_extractor,
)

print("\n🚀 학습 시작! (잠시만 기다려주세요...)")
trainer.train()

# ==========================================
# 6. 저장
# ==========================================
print(f"\n💾 모델 저장 중... ({OUTPUT_DIR})")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("🎉 학습 완료! 이제 03_inference.py를 만들어 실행해보세요.")
