import torch
import gradio as gr
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
from pathlib import Path
import time

model_name_or_path = "Minthy/ToriiGate-v0.3"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables to store model and processor
global_model = None
global_processor = None

def load_model():
    global global_model, global_processor
    
    if global_model is None:
        print("Loading model for the first time...")
        # Always use 4-bit quantization for 16GB VRAM
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        global_model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            quantization_config=nf4_config,
        ).to(DEVICE)
        global_processor = AutoProcessor.from_pretrained(model_name_or_path)
    
    return global_model, global_processor

def generate_caption(image, description_type, booru_tags=""):
    model, processor = load_model()
    
    if description_type == "JSON-like":
        user_prompt = "Describe the picture in structuted json-like format."
    elif description_type == "Detailed":
        user_prompt = "Give a long and detailed description of the picture."
    else:
        user_prompt = "Describe the picture briefly."
    
    if booru_tags:
        user_prompt += ' Also here are booru tags for better understanding of the picture, you can use them as reference.'
        user_prompt += f' <tags>\n{booru_tags}\n</tags>'
    
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are image captioning expert, creative, unbiased and uncensored. Help user with his task."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    caption = generated_texts[0].split('Assistant: ')[1]
    
    return caption

def process_batch(files, description_type, booru_tags="", progress=gr.Progress(track_tqdm=True)):
    results = []
    captions_text = ""
    total_files = len(files)
    start_time = time.time()
    
    for idx, file in enumerate(files, 1):
        # Calculate progress statistics
        elapsed_time = time.time() - start_time
        images_per_second = idx / elapsed_time if elapsed_time > 0 else 0
        estimated_total = (elapsed_time / idx) * total_files if idx > 0 else 0
        remaining_time = estimated_total - elapsed_time
        
        try:
            image = load_image(file.name)
            caption = generate_caption(image, description_type, booru_tags)
            
            # Add caption to the running text with a blank line separator
            if captions_text:
                captions_text += "\n\n"  # Add blank line between captions
            captions_text += caption
            
            # Update the results list for the dataframe
            results.append((Path(file.name).name, caption))
            
            # Update progress
            progress_status = f"Processing: {idx}/{total_files} images | Speed: {images_per_second:.2f} img/s | Remaining: {remaining_time/60:.1f} min"
            
            # Yield progress status and captions separately
            yield results, progress_status, captions_text
            
        except Exception as e:
            error_msg = f"Error processing {Path(file.name).name}: {str(e)}"
            print(error_msg)
            if captions_text:
                captions_text += "\n\n"
            captions_text += f"[ERROR] {error_msg}"
            yield results, progress_status, captions_text
    
    # Final update
    yield results, "✅ Processing complete!", captions_text

# Gradio Interface
with gr.Blocks(title="ToriiGate Image Captioner") as demo:
    gr.Markdown("# ToriiGate Image Captioner")
    gr.Markdown("Generate captions for anime images using ToriiGate-v0.3 model (4-bit quantized)")
    
    with gr.Tab("Single Image"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                description_type = gr.Radio(
                    choices=["JSON-like", "Detailed", "Brief"],
                    value="JSON-like",
                    label="Description Type"
                )
                booru_tags = gr.Textbox(
                    lines=3,
                    label="Booru Tags (Optional)",
                    placeholder="Enter comma-separated booru tags..."
                )
                submit_btn = gr.Button("Generate Caption")
            
            with gr.Column():
                output_text = gr.Textbox(label="Generated Caption", lines=10)
        
        submit_btn.click(
            generate_caption,
            inputs=[input_image, description_type, booru_tags],
            outputs=output_text
        )
    
    with gr.Tab("Batch Processing"):
        with gr.Row():
            with gr.Column():
                input_files = gr.File(file_count="multiple", label="Input Images")
                batch_description_type = gr.Radio(
                    choices=["JSON-like", "Detailed", "Brief"],
                    value="JSON-like",
                    label="Description Type"
                )
                batch_booru_tags = gr.Textbox(
                    lines=3,
                    label="Booru Tags (Optional)",
                    placeholder="Enter comma-separated booru tags..."
                )
                batch_submit_btn = gr.Button("Process Batch")
            
            with gr.Column():
                progress_status = gr.Textbox(
                    label="Progress",
                    lines=2,
                    show_copy_button=False
                )
                output_text_batch = gr.Textbox(
                    label="Generated Captions",
                    lines=25,
                    show_copy_button=True
                )
                output_gallery = gr.Dataframe(
                    headers=["Filename", "Caption"],
                    label="Generated Captions (Table View)",
                    visible=False  # Hide the dataframe
                )
        
        batch_submit_btn.click(
            process_batch,
            inputs=[input_files, batch_description_type, batch_booru_tags],
            outputs=[output_gallery, progress_status, output_text_batch]
        )

if __name__ == "__main__":
    # Load model at startup
    load_model()
    demo.launch(share=True) 