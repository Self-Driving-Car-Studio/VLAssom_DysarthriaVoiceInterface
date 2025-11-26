import os
import torch
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# ==========================================
# ⚙️ 설정
# ==========================================
# 1. 기본 모델과 학습된 어댑터(LoRA) 경로
BASE_MODEL = "openai/whisper-small"
ADAPTER_PATH = "../models/whisper-finetuned-v1"

# 2. 녹음 설정
SR = 16000  # Whisper는 무조건 16kHz
RECORD_SECONDS = 3  # 한 번에들을 시간

# ==========================================
# 1. 모델 로딩 (학습된 결과 합치기)
# ==========================================
print("⏳ 모델을 불러오는 중입니다... (시간이 조금 걸립니다)")

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 실행 장치: {device}")

# 기본 프로세서 & 모델 로드
processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="Korean", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, device_map=device)

# 🌟 핵심: 내가 학습시킨 LoRA 어댑터를 기본 모델에 장착!
if os.path.exists(ADAPTER_PATH):
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ 학습된 맞춤형 모델(LoRA)이 성공적으로 적용되었습니다!")
else:
    print("⚠️ 경고: 학습된 모델을 찾을 수 없습니다. 기본 모델로 동작합니다.")

# ==========================================
# 2. 추론 및 로봇 제어 함수
# ==========================================
def robot_action(text):
    """인식된 텍스트에 따라 로봇 동작을 결정하는 함수"""
    print(f"\n🤖 [인식 결과]: '{text}'")
    
    if "비타민" in text and "줘" in text:
        print("   └─ 🦾 동작: VLAssom이 비타민을 잡아서 가져옵니다.")
    elif "타이레놀" in text and "줘" in text :
        print("   └─ 🦾 동작: VLAssom이 타이레놀을 잡아서 가져옵니다.")
    elif "연필" in text and "줘" in text:
        print("   └─ 🦾 동작: VLAssom이 연필을 잡아서 가져옵니다.")
    else:
        print("   └─ ❓ 동작: (정의되지 않은 명령어입니다)")

def transcribe_audio(audio_data):
    """오디오 데이터를 텍스트로 변환"""
    # 1. 전처리
    input_features = processor(
        audio_data, 
        sampling_rate=SR, 
        return_tensors="pt"
    ).input_features.to(device)

    # 2. 추론 (생성)
    with torch.no_grad():
        generated_ids = model.generate(input_features, language="korean")

    # 3. 디코딩 (숫자 -> 글자)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription.strip()

# ==========================================
# 3. 메인 실행 루프
# ==========================================
print("\n" + "="*50)
print("🎤 VLAssom 음성 제어 인터페이스 시작")
print("="*50)

try:
    while True:
        input("\n⌨️ 엔터(Enter)를 누르면 3초간 듣습니다... (종료: Ctrl+C)")
        
        # 1. 녹음
        print("🔴 듣고 있습니다...")
        recording = sd.rec(int(RECORD_SECONDS * SR), samplerate=SR, channels=1)
        sd.wait()
        print("✅ 처리 중...")

        # 2. 차원 변환 (Whisper 입력 규격에 맞춤)
        audio_data = recording.flatten()

        # 3. 텍스트 변환
        result_text = transcribe_audio(audio_data)

        # 4. 로봇 동작 실행
        robot_action(result_text)

except KeyboardInterrupt:
    print("\n👋 프로그램을 종료합니다.")
