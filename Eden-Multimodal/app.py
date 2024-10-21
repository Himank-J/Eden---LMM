import gradio as gr
import torch
from models.multimodel_phi import MultimodalPhiModel
from utils.audio_processing import transcribe_speech
from utils.image_processing import getImageArray
from utils.text_processing import getStringAfter, getAnswerPart, getInputs
from config import device, model_location, base_phi_model, tokenizer

model = MultimodalPhiModel.from_pretrained(model_location).to(device)

def output_parser(image_path, audio_path, context_text, question, max_length=3):
    answerPart = ""
    speech_text = ""
    if image_path is not None:
        for i in range(max_length):
            start_tokens, end_tokens, image_features, attention_mask = getInputs(image_path, question, answer=answerPart)
            output = model(start_tokens, end_tokens, image_features, attention_mask, labels=None)
            tokens = output.logits.argmax(dim=-1)
            output = tokenizer.decode(tokens[0], skip_special_tokens=True)
            answerPart = getAnswerPart(output)
        print("Answerpart:", answerPart)

    if audio_path is not None:
        speech_text = transcribe_speech(audio_path)
        print("Speech Text:", speech_text)

    if (question is None) or (question == ""):
        question = " Describe the objects and their relationships in 1 sentence."

    input_text = (
        "<|system|>\n Please understand the context "
        "and answer the question in 1 or 2 summarized sentences.\n"
        f"<|end|>\n<|user|>\n<|context|> {answerPart} \n {speech_text} \n {context_text} "
        f"\n<|question|>: {question} \n<|end|>\n<|assistant|>\n"
    )
    print("input_text:", input_text)
    tokens = tokenizer(input_text, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    start_tokens = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    output_text = tokenizer.decode(
        base_phi_model.generate(start_tokens, attention_mask=attention_mask, max_length=1024, do_sample=False, pad_token_id=tokenizer.pad_token_id)[0],
        skip_special_tokens=True
    )

    output_text = getStringAfter(output_text, question).strip()
    return output_text

# Gradio interface setup
title = "Created Fine Tuned MultiModal model"
description = "Test the fine tuned multimodal model created using clip, phi3.5 mini instruct, whisper models"
 
def process_chat_input(history, message, audio):
    image_path = next((file for file in message["files"] if file.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'gif']), None)
    audio_path = next((file for file in message["files"] if file.split('.')[-1].lower() in ['mp3', 'wav', 'ogg']), None) or audio
    question = message["text"]
 
    response = output_parser(image_path, audio_path, "", question)
 
    if image_path:
        history.append({"role": "user", "content": {"path": image_path}})
    if audio_path:
        history.append({"role": "user", "content": {"path": audio_path}})
    if question:
        history.append({"role": "user", "content": question})
 
    history.append({"role": "assistant", "content": ""})
    for char in response:
        history[-1]["content"] += char
        yield history, ""
 
custom_theme = gr.themes.Base(
    primary_hue="gray",
    secondary_hue="gray",
    neutral_hue="gray",
    font=["Helvetica", "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#000000",
    body_text_color="#ffffff",
    color_accent_soft="*neutral_600",
    background_fill_primary="#111111",
    background_fill_secondary="#222222",
    border_color_accent="*neutral_700",
    button_primary_background_fill="*neutral_800",
    button_primary_text_color="#ffffff",
    # Add these lines to ensure all text is white
    block_title_text_color="#ffffff",
    block_label_text_color="#ffffff"
)
 
with gr.Blocks(theme=custom_theme) as demo:
    with gr.Row():
       gr.Markdown("# Eden")
    gr.Markdown("Chat with the fine-tuned multimodal Eden using text, audio, or image inputs.")
 
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        height=450,
        type="messages"
    )
 
    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter your message, upload an image, or upload an audio file...",
        show_label=False,
        file_types=["image", "audio"],
        container=False,
        scale=3,
        lines=1
    )
 
    gr.Markdown("Or record a message:")
    audio_input = gr.Audio(type="filepath", sources=["microphone", "upload"])
 
    chat_input.submit(
        process_chat_input,
        [chatbot, chat_input, audio_input],
        [chatbot, chat_input]
    ).then(lambda: gr.MultimodalTextbox(interactive=True, lines=1), None, [chat_input])
 
    gr.Examples(
        examples=[
            "Describe the objects in the image.",
            "What can you hear in the audio?",
            "Summarize the context provided.",
        ],
        inputs=chat_input,
    )
 
demo.launch(debug=True)