# 🦾 VLAssom Dysarthria Voice Interface
> **Personalized Speech Recognition for Dysarthria Patients** > 구음장애 환자를 위한 맞춤형 음성 인식 및 로봇 제어 인터페이스

## 📖 프로젝트 소개 (Project Overview)
이 프로젝트는 **구음장애(Dysarthria)** 를 가진 사용자의 불분명한 발음을 정확하게 인식하여, **보조 로봇(VLAssom)** 을 제어하기 위해 개발되었습니다.

기존의 범용 음성 인식 모델(Google STT, Siri 등)은 구음장애 환자의 발음 패턴을 인식하지 못하는 한계가 있습니다. 이를 해결하기 위해 **OpenAI Whisper** 모델을 사용자의 목소리 데이터로 **파인튜닝(Fine-tuning, LoRA)** 하여, 소량의 데이터만으로도 개인화된 음성 인식 모델을 구축했습니다.

## ✨ 주요 기능 (Key Features)
* **🛠️ 자동 데이터 수집기 (`Recorder`):** 반복적인 음성 데이터 수집을 돕는 자동화 스크립트 (파일명 자동 생성 및 CSV 라벨링).
* **🧠 효율적인 파인튜닝 (`LoRA Training`):** `PEFT(LoRA)` 기술을 적용하여, 일반 GPU 환경에서도 빠르고 가볍게 Whisper 모델 학습 가능.
* **⚡ 실시간 추론 및 제어 (`Inference`):** 학습된 모델을 로드하여 마이크 입력을 실시간 텍스트로 변환하고 로봇 제어 명령으로 매핑.

## 📂 폴더 구조 (Directory Structure)
```bash
VLAssom-Voice-Interface/
├── dataset/                 # 학습 데이터셋
│   ├── raw_audio/           # 녹음된 WAV 파일들
│   └── metadata.csv         # 데이터 라벨링 정답지
├── models/                  # 학습된 모델 저장소
├── src/                     # 소스 코드
│   ├── 01_recorder.py       # 데이터 수집 도구
│   ├── 02_finetune.py       # Whisper 모델 학습 (LoRA)
│   └── 03_inference.py      # 실시간 추론 및 로봇 제어
└── requirements.txt         # 필요 라이브러리 목록
