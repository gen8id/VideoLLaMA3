"""
VideoLLaMA3 모델 다운로드 스크립트
컨테이너 실행 후 한 번만 실행하면 됩니다.
"""

import os
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download

def download_model(model_name="DAMO-NLP-SG/VideoLLaMA3-7B", cache_dir="/workspace/models"):
    """
    모델을 다운로드하여 캐시 디렉토리에 저장
    
    Args:
        model_name: HuggingFace 모델 이름
        cache_dir: 모델을 저장할 디렉토리
    """
    
    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print("This may take 10-20 minutes depending on your internet speed...")
    print("-" * 80)
    
    # 환경 변수 설정
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    
    try:
        # 방법 1: transformers 라이브러리로 다운로드 (권장)
        print("\n[1/2] Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        print("✓ Model weights downloaded successfully")
        
        print("\n[2/2] Downloading processor/tokenizer...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        print("✓ Processor downloaded successfully")
        
        # 메모리 해제
        del model
        del processor
        
        print("\n" + "=" * 80)
        print("✅ Download completed successfully!")
        print(f"Model cached at: {cache_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during download: {str(e)}")
        print("\nTrying alternative method...")
        
        # 방법 2: huggingface_hub로 전체 다운로드
        try:
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                resume_download=True,
            )
            print("✓ Download completed using alternative method")
        except Exception as e2:
            print(f"❌ Alternative method also failed: {str(e2)}")
            raise


def download_multiple_models(models, cache_dir="/workspace/models"):
    """
    여러 모델을 한 번에 다운로드
    
    Args:
        models: 다운로드할 모델 이름 리스트
        cache_dir: 저장 디렉토리
    """
    
    print(f"Downloading {len(models)} models...")
    print("=" * 80)
    
    for i, model_name in enumerate(models, 1):
        print(f"\n[Model {i}/{len(models)}]")
        try:
            download_model(model_name, cache_dir)
        except Exception as e:
            print(f"Failed to download {model_name}: {str(e)}")
            continue
    
    print("\n" + "=" * 80)
    print("All downloads completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download VideoLLaMA3 models")
    parser.add_argument(
        "--model", 
        type=str, 
        default="DAMO-NLP-SG/VideoLLaMA3-7B",
        choices=[
            "DAMO-NLP-SG/VideoLLaMA3-7B",
            "DAMO-NLP-SG/VideoLLaMA3-2B",
            "DAMO-NLP-SG/VideoLLaMA3-7B-Image",
            "DAMO-NLP-SG/VideoLLaMA3-2B-Image",
        ],
        help="Model to download"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="/workspace/models",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    
    args = parser.parse_args()
    
    if args.all:
        models = [
            "DAMO-NLP-SG/VideoLLaMA3-7B",
            "DAMO-NLP-SG/VideoLLaMA3-2B",
        ]
        download_multiple_models(models, args.cache_dir)
    else:
        download_model(args.model, args.cache_dir)