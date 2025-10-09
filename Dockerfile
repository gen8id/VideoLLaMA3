FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

# 환경 변수 설정
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# VideoLLaMA3 저장소 클론
RUN git clone https://github.com/gen8id/VideoLLaMA3.git && \
    cd VideoLLaMA3

WORKDIR /workspace/VideoLLaMA3

# 의존성 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# flash-attention 설치 (시간 오래 걸림)
# Flash-attn pinned to a compatible version
RUN pip install flash-attn==2.7.3 --no-build-isolation --upgrade

# 추가 필수 패키지 설치 (torchvision 버전 맞춤)
RUN pip install --no-cache-dir \
    torchvision==0.19.0 \
    transformers==4.46.3 \
    accelerate==1.0.1 \
    decord \
    ffmpeg-python \
    imageio \
    opencv-python

# 모델 다운로드용 디렉토리 생성
RUN mkdir -p /workspace/models /workspace/videos /workspace/outputs

# 환경 변수 설정 (HuggingFace 캐시)
ENV HF_HOME=/workspace/models \
    TRANSFORMERS_CACHE=/workspace/models \
    HF_HUB_CACHE=/workspace/models

#COPY ./scripts/download-model.py /workspace/VideoLLaMA3/scripts
#COPY ./scripts/inference.py /workspace/VideoLLaMA3/scripts
#COPY ./scripts/gradio_simple_by_ai_companion.py /workspace/VideoLLaMA3/scripts

# 모델 사전 다운로드 (선택사항 - 주석 해제하여 사용)
#RUN python -c "from transformers import AutoModelForCausalLM, AutoProcessor; \
#    AutoModelForCausalLM.from_pretrained('DAMO-NLP-SG/VideoLLaMA3-7B', trust_remote_code=True); \
#    AutoProcessor.from_pretrained('DAMO-NLP-SG/VideoLLaMA3-7B', trust_remote_code=True)"

# 포트 노출 (Gradio UI 데모)
EXPOSE 80
# 포트 노출 (Gradio API 데모)
EXPOSE 8080

# 기본 명령어
CMD ["/bin/bash"]