import torch
import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor

def summarize_video(video_path, output_path, model_path="DAMO-NLP-SG/VideoLLaMA3-7B",
                    fps=1, max_frames=180, max_new_tokens=512):
    """
    비디오를 분석하고 요약을 텍스트 파일로 저장
    
    Args:
        video_path: 입력 비디오 경로
        output_path: 출력 텍스트 파일 경로
        model_path: 모델 경로 (HuggingFace 또는 로컬)
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 모델 로드
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    print(f"Processing video: {video_path}")
    
    # 비디오 요약 생성
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that analyzes videos."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": {
                        "video_path": video_path, 
                        "fps": fps,  # 1 FPS로 샘플링
                        "max_frames": max_frames  # 최대 180 프레임
                    }
                },
                {
                    "type": "text", 
                    "text": """Please provide a comprehensive summary of this video including:
1. Main subjects and objects in the video
2. Key actions and events that occur
3. Overall context and theme
4. Any notable details or important moments

Provide the summary in a clear, structured format."""
                },
            ]
        },
    ]
    
    # 입력 준비
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v 
        for k, v in inputs.items()
    }
    
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    
    # 추론 실행
    print("Generating summary...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    # 결과 디코딩
    response = processor.batch_decode(
        output_ids, 
        skip_special_tokens=True
    )[0].strip()
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Video: {video_path}\n")
        f.write("="*80 + "\n\n")
        f.write(response)
        f.write("\n")
    
    print(f"Summary saved to: {output_path}")
    return response


def batch_process(video_dir, output_dir, model_path="DAMO-NLP-SG/VideoLLaMA3-7B",
                  fps=1, max_frames=180, max_new_tokens=512):
    """
    여러 비디오를 배치 처리
    
    Args:
        video_dir: 비디오가 있는 디렉토리
        output_dir: 결과를 저장할 디렉토리
        model_path: 모델 경로
    """
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    # 모델은 한 번만 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    # 각 비디오 처리
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
        
        output_file = Path(output_dir) / f"{video_file.stem}_summary.txt"
        
        try:
            # 동일한 로직으로 처리
            conversation = [
                {"role": "system", "content": "You are a helpful assistant that analyzes videos."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video", 
                            "video": {
                                "video_path": str(video_file), 
                                "fps": fps,
                                "max_frames": max_frames
                            }
                        },
                        {
                            "type": "text", 
                            "text": "Provide a detailed summary of this video including main subjects, actions, and context."
                        },
                    ]
                },
            ]
            
            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
            }
            
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            
            with torch.autocast(device_type=device, dtype=torch.float16):
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            
            response = processor.batch_decode(
                output_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            # 저장
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Video: {video_file.name}\n")
                f.write("="*80 + "\n\n")
                f.write(response)
                f.write("\n")
            
            print(f"✓ Saved to: {output_file}")
            
        except Exception as e:
            print(f"✗ Error processing {video_file.name}: {str(e)}")
            continue

    # GPU 메모리 해제 (필요 시)
    # del model
    # torch.cuda.empty_cache()
    # print("All videos processed and GPU memory cleared.")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VideoLLaMA3 Video Summary Tool")
    parser.add_argument("--video", type=str, help="Single video file path")
    parser.add_argument("--video_dir", type=str, help="Directory containing videos for batch processing")
    parser.add_argument("--output", type=str, default="../outputs/summary.txt", help="Output text file path")
    parser.add_argument("--output_dir", type=str, default="../outputs", help="Output directory for batch processing")
    parser.add_argument("--model", type=str, default="DAMO-NLP-SG/VideoLLaMA3-7B", help="Model path")
    
    args = parser.parse_args()
    
    if args.video:
        # 단일 비디오 처리
        summarize_video(args.video, args.output, args.model)
    elif args.video_dir:
        # 배치 처리
        batch_process(args.video_dir, args.output_dir, args.model)
    else:
        print("Please specify either --video or --video_dir")
        parser.print_help()