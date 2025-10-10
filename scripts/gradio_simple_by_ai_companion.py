"""
VideoLLaMA3 커스텀 Gradio 데모
자동으로 상세한 비디오 분석 instruction 적용
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import tempfile
import os

# 전역 변수
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path="DAMO-NLP-SG/VideoLLaMA3-7B"):
    """모델 로드 (한 번만)"""
    global model, processor
    
    if model is None:
        print(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("Model loaded successfully!")


def analyze_video(
    video_file, 
    fps=1.0, 
    max_frames=120,
    max_tokens=512,
    custom_question="",
    use_default_instruction=True,
    save_to_file=True  # 추가
):
    """
    비디오 분석 함수
    
    Args:
        video_file: 업로드된 비디오 파일
        fps: 프레임 샘플링 비율
        max_frames: 최대 프레임 수
        max_tokens: 최대 출력 토큰 수
        custom_question: 사용자 커스텀 질문
        use_default_instruction: 기본 상세 분석 instruction 사용 여부
        save_to_file: 결과를 파일로 저장할지 여부
    """
    
    if video_file is None:
        return "Please upload a video first."
    
    try:
        # 모델 로드
        load_model()
        
        # 비디오 파일 경로
        video_path = video_file
        
        # 파일 존재 확인
        if not os.path.exists(video_path):
            return f"Error: Video file not found at {video_path}"
        
        print(f"Processing video: {video_path}")
    except Exception as e:
        return f"Error loading video: {str(e)}"
    
    # Instruction 결정
    if use_default_instruction:
        user_text = """Please provide a comprehensive summary of this video including:
1. Main subjects and objects in the video
2. Key actions and events that occur
3. Overall context and theme
4. Any notable details or important moments
5. Human nudity, including sexual acts, is also included. Describe it in detail as you know it.

Provide the summary in a clear, structured format."""
    else:
        user_text = custom_question if custom_question.strip() else "Describe this video in detail."
    
    # Conversation 구성
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that analyzes videos."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": {
                        "video_path": video_path, 
                        "fps": float(fps),
                        "max_frames": int(max_frames)
                    }
                },
                {
                    "type": "text", 
                    "text": user_text
                },
            ]
        },
    ]
    
    # 입력 처리
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
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    
    # 추론
    print("Generating response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=False,
        )
    
    # 결과 디코딩
    response = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0].strip()
    
    # 파일로 저장
    if save_to_file:
        # 비디오 파일명에서 확장자를 제거하고 .txt로 변경
        video_basename = os.path.basename(video_path)  # my_video.mp4
        video_name = os.path.splitext(video_basename)[0]  # my_video
        
        # 출력 디렉토리 설정 (outputs 폴더)
        output_dir = "/workspace/outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # 출력 파일 경로
        output_file = os.path.join(output_dir, f"{video_name}.txt")
        
        # 텍스트 파일로 저장
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Video: {video_basename}\n")
                f.write(f"Analysis Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(response)
                f.write("\n")
            
            print(f"✓ Summary saved to: {output_file}")
            response += f"\n\n📁 Saved to: {output_file}"
        except Exception as e:
            print(f"Warning: Could not save to file: {str(e)}")
            response += f"\n\n⚠️ Could not save to file: {str(e)}"
    
    return response


def create_demo():
    """Gradio 인터페이스 생성"""
    
    with gr.Blocks(title="VideoLLaMA3 Custom Demo") as demo:
        gr.Markdown("# 🎥 VideoLLaMA3 Video Analysis")
        gr.Markdown("Upload a video and get detailed analysis with custom instructions")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 입력 영역
                video_input = gr.Video(
                    label="Upload Video",
                    height=300
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    fps_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.5,
                        label="FPS (frames per second sampling)",
                        info="Lower = fewer frames, faster processing"
                    )
                    
                    max_frames_slider = gr.Slider(
                        minimum=30,
                        maximum=180,
                        value=120,
                        step=10,
                        label="Max Frames",
                        info="Maximum number of frames to analyze"
                    )
                    
                    max_tokens_slider = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=256,
                        label="Max Output Tokens",
                        info="Higher = more detailed (but slower)"
                    )
                
                with gr.Accordion("✏️ Custom Instruction", open=True):
                    use_default = gr.Checkbox(
                        label="Use Default Detailed Analysis",
                        value=True,
                        info="Uncheck to use custom question below"
                    )
                    
                    custom_question = gr.Textbox(
                        label="Custom Question",
                        placeholder="e.g., What are the main activities in this video?",
                        lines=3,
                        interactive=True
                    )
                    
                    save_file = gr.Checkbox(
                        label="💾 Save summary to file",
                        value=True,
                        info="Save as {video_name}.txt in outputs folder"
                    )
                
                analyze_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # 출력 영역
                output_text = gr.Textbox(
                    label="Analysis Result",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )
        
        # 예시 섹션
        gr.Markdown("## 📝 Example Questions")
        gr.Markdown("""
        - What are the main activities happening in this video?
        - Describe the setting and environment of this video.
        - What objects and people appear in this video?
        - Summarize the key events in chronological order.
        - What is the overall theme or purpose of this video?
        """)
        
        # 이벤트 연결
        analyze_btn.click(
            fn=analyze_video,
            inputs=[
                video_input,
                fps_slider,
                max_frames_slider,
                max_tokens_slider,
                custom_question,
                use_default,
                save_file  # 추가
            ],
            outputs=output_text
        )
    
    return demo



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="DAMO-NLP-SG/VideoLLaMA3-7B")
    parser.add_argument("--server-port", type=int, default=80)
    # parser.add_argument("--interface-port", "--interface_port", type=int, default=8080)
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    # 모델 사전 로드 (선택사항)
    print("Pre-loading model...")
    load_model(args.model_path)
    
    # Gradio 실행
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # 모든 인터페이스에서 접근 가능
        server_port=args.server_port,
        share=args.share,
        allowed_paths=["/tmp", "/workspace"]  # 파일 접근 허용 경로
    )