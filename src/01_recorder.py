import sounddevice as sd
import scipy.io.wavfile as wav
import pandas as pd
import os
import time

# ==========================================
# ⚙️ 설정 (내 프로젝트 환경에 맞게 수정 가능)
# ==========================================
SR = 16000                # Whisper 모델 권장 샘플링 레이트 (16kHz)
RECORD_SECONDS = 3        # 파일당 녹음 시간 (초)
DATA_ROOT = "../dataset"  # 데이터 저장 최상위 폴더 (src 폴더 상위)
AUDIO_FOLDER = os.path.join(DATA_ROOT, "raw_audio")
CSV_FILE = os.path.join(DATA_ROOT, "metadata.csv")

# 폴더 생성
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

def get_next_index(folder, label):
    """
    폴더를 뒤져서 해당 라벨(label)의 다음 번호를 찾아내는 함수
    예: water_001.wav, water_002.wav가 있으면 -> 3 반환
    """
    files = [f for f in os.listdir(folder) if f.startswith(label) and f.endswith(".wav")]
    if not files:
        return 1
    
    # 파일명에서 숫자만 추출해서 가장 큰 수 찾기
    indices = []
    for f in files:
        try:
            # "water_001.wav" -> "001" -> 1
            idx = int(f.split('_')[-1].split('.')[0])
            indices.append(idx)
        except:
            continue
            
    return max(indices) + 1 if indices else 1

def update_csv(filename, sentence):
    """CSV 파일에 새로운 데이터 한 줄 추가"""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=["file_name", "sentence"])
    
    new_data = {"file_name": filename, "sentence": sentence}
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")

# ==========================================
# 🎤 메인 녹음 프로그램
# ==========================================
print("\n" + "="*50)
print(f"🤖 VLAssom 맞춤형 음성 데이터 수집기")
print(f"📂 저장 위치: {AUDIO_FOLDER}")
print("="*50)

try:
    while True:
        print("\n📝 새로운 녹음 세트 시작 (종료하려면 Ctrl+C)")
        
        # 1. 정보 입력
        # 파일명에 쓸 영어 라벨 (예: water)
        label_eng = input("1. 파일명 라벨 (영어, 예: water): ").strip()
        if not label_eng: continue
        
        # 정답지(CSV)에 들어갈 텍스트 (예: 물 줘)
        sentence_kor = input(f"2. 정답 텍스트 (한국어, 예: 물 줘): ").strip()
        
        # 반복 횟수
        try:
            repeat_count = int(input("3. 몇 번 반복해서 녹음할까요? (숫자, 예: 5): "))
        except:
            repeat_count = 1

        print("-" * 30)
        print(f"📢 '{sentence_kor}' ({label_eng}) -> {repeat_count}회 녹음을 시작합니다.")
        input("⌨️ 준비되면 엔터(Enter)를 누르세요...")

        # 2. 반복 녹음 시작
        for i in range(repeat_count):
            # 다음 번호 자동 계산
            current_idx = get_next_index(AUDIO_FOLDER, label_eng)
            filename = f"{label_eng}_{current_idx:03d}.wav" # 예: water_001.wav
            filepath = os.path.join(AUDIO_FOLDER, filename)

            print(f"\n[{i+1}/{repeat_count}] 🔴 녹음 중... ({filename})")
            
            # 녹음 수행
            recording = sd.rec(int(RECORD_SECONDS * SR), samplerate=SR, channels=1)
            sd.wait() # 녹음 끝날 때까지 대기
            
            # 파일 저장
            wav.write(filepath, SR, recording)
            
            # CSV 업데이트
            update_csv(filename, sentence_kor)
            
            print(f"✅ 저장 완료! (잠시 대기...)")
            time.sleep(1) # 연속 녹음 시 1초 숨 고르기

        print(f"\n🎉 {label_eng} 세트 녹음 완료!")

except KeyboardInterrupt:
    print("\n\n💾 프로그램을 종료합니다. 수고하셨습니다!")