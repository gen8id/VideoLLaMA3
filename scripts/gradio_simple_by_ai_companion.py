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


def analyze_video(video_file, fps, max_frames, max_tokens, custom_question, use_default, save_file):
    """비디오 분석 메인 함수"""

    if video_file is None:
        return "❌ Please upload a video first."

    try:
        # 모델 로드
        load_model()

        video_path = video_file

        if not os.path.exists(video_path):
            return f"❌ Error: Video file not found at {video_path}"

        print(f"📹 Processing video: {video_path}")

        # Instruction 결정
        if use_default:
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
                    {"type": "text", "text": user_text},
                ]
            },
        ]

        # 입력 처리
        print("🔄 Processing inputs...")
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
        print("🤖 Generating response...")
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(max_tokens),
                    do_sample=False,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return "❌ GPU Out of Memory! Try reducing max_frames or max_tokens."
            raise

        # 결과 디코딩
        response = processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0].strip()

        print("✅ Analysis complete!")

        # 파일 저장
        if save_file:
            try:
                video_basename = os.path.basename(video_path)
                video_name = os.path.splitext(video_basename)[0]

                output_dir = "/workspace/outputs"
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(output_dir, f"{video_name}.txt")

                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Video: {video_basename}\n")
                    f.write(f"Analysis Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(response)
                    f.write("\n")

                print(f"💾 Saved to: {output_file}")
                response += f"\n\n📁 Saved to: {output_file}"
            except Exception as e:
                print(f"⚠️ Could not save file: {str(e)}")
                response += f"\n\n⚠️ Could not save to file: {str(e)}"

        return response

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


def create_demo():
    """Gradio 인터페이스 생성"""

    with gr.Blocks(
            title="VideoLLaMA3 Analysis",
            theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# 🎥 VideoLLaMA3 Video Analysis")
        gr.Markdown("Upload a video and get detailed AI-powered analysis")

        with gr.Row():
            # 왼쪽: 입력
            with gr.Column(scale=1):
                video_input = gr.Video(label="📹 Upload Video")

                with gr.Accordion("⚙️ Settings", open=False):
                    fps_input = gr.Slider(
                        0.5, 2.0, value=1.0, step=0.5,
                        label="FPS Sampling",
                        info="Frames per second to sample"
                    )

                    frames_input = gr.Slider(
                        30, 180, value=90, step=10,
                        label="Max Frames",
                        info="Lower = less memory usage"
                    )

                    tokens_input = gr.Slider(
                        256, 2048, value=768, step=256,
                        label="Max Output Tokens",
                        info="Higher = more detailed output"
                    )

                with gr.Accordion("✏️ Instruction", open=True):
                    default_checkbox = gr.Checkbox(
                        label="Use default detailed analysis",
                        value=True
                    )

                    custom_text = gr.Textbox(
                        label="Custom question",
                        placeholder="Enter your question about the video...",
                        lines=3
                    )

                    save_checkbox = gr.Checkbox(
                        label="💾 Save to file",
                        value=True
                    )

                submit_btn = gr.Button(
                    "🚀 Analyze Video",
                    variant="primary",
                    size="lg"
                )

            # 오른쪽: 출력
            with gr.Column(scale=1):
                output_box = gr.Textbox(
                    label="📄 Analysis Result",
                    lines=25,
                    show_copy_button=True
                )

        gr.Markdown("### 💡 Tips")
        gr.Markdown("""
        - **Short videos** (< 2 min) work best
        - Reduce **max_frames** if you get memory errors
        - Use **custom question** for specific analysis
        """)

        # 이벤트 바인딩
        submit_btn.click(
            fn=analyze_video,
            inputs=[
                video_input,
                fps_input,
                frames_input,
                tokens_input,
                custom_text,
                default_checkbox,
                save_checkbox
            ],
            outputs=output_box
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="DAMO-NLP-SG/VideoLLaMA3-7B")
    parser.add_argument("--server-port", type=int, default=7860)  # ✅ 80 대신 7860 권장
    parser.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()

    print("Pre-loading model...")
    load_model(args.model_path)

    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",   # ✅ 모든 인터페이스에서 접근 허용
        server_port=args.server_port,
        share=False,              # ✅ localhost 접근 불가 환경에서 필수
        allowed_paths=["/tmp", "/workspace"],  # ✅ 안전 경로 지정
        show_api=True,          # 선택: API 엔드포인트 숨김
    )