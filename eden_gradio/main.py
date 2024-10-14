import torch
import gradio as gr
from models.multimodal_phi_model import MultimodalPhiModel
from utils.audio_processing import transcribe_speech
from utils.image_processing import getImageArray, get_clip_embeddings
from utils.text_processing import getStringAfter, getAnswerPart
from config import device, model_location

model = MultimodalPhiModel.from_pretrained(model_location).to(device)
base_phi_model = model.base_phi_model.to(device)

def getInputs(image_path, question, answer=""):
    image_features = None
    num_image_tokens = 0

    if image_path is not None:
        image = getImageArray(image_path)
        image_features = get_clip_embeddings(image)
        image_features = torch.stack([image_features])
        num_image_tokens = image_features.shape[1]

    start_text = f"<|system|>\nYou are an assistant good at understanding the context.<|end|>\n<|user|>\n "
    end_text = f" .\n  Describe the objects and their relationship from the context. <|end|>\n<|assistant|>\n {answer}"

    start_tokens = model.tokenizer(start_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    end_tokens = model.tokenizer(end_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    start_input_ids = start_tokens['input_ids']
    start_attention_mask = start_tokens['attention_mask']
    end_input_ids = end_tokens['input_ids']
    end_attention_mask = end_tokens['attention_mask']

    if image_path is not None:
        attention_mask = torch.cat([start_attention_mask, torch.ones((1, num_image_tokens), dtype=torch.long), end_attention_mask], dim=1)
    else:
        attention_mask = torch.cat([start_attention_mask, end_attention_mask], dim=1)

    return start_input_ids, end_input_ids, image_features, attention_mask

def generateOutput(image_path, audio_path, context_text, question, max_length=3):
    answerPart = ""
    speech_text = ""
    if image_path is not None:
        for i in range(max_length):
            start_tokens, end_tokens, image_features, attention_mask = getInputs(image_path, question, answer=answerPart)
            output = model(start_tokens, end_tokens, image_features, attention_mask, labels=None)
            tokens = output.logits.argmax(dim=-1)
            output = model.tokenizer.decode(tokens[0], skip_special_tokens=True)
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
    tokens = model.tokenizer(input_text, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    start_tokens = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    output_text = model.tokenizer.decode(
        base_phi_model.generate(start_tokens, attention_mask=attention_mask, max_length=1024, do_sample=False, pad_token_id=model.tokenizer.pad_token_id)[0],
        skip_special_tokens=True
    )

    output_text = getStringAfter(output_text, question).strip()
    return output_text

title = "Created Fine Tuned MultiModal model"
description = "Test the fine tuned multimodal model created using clip, phi3.5 mini instruct, whisper models"

def process_input(history, message, audio):
    image_path = None
    audio_path = None
    context_text = ""
    question = ""

    if message["files"]:
        image_path = message["files"][0]  

    if message["text"]:
        question = message["text"]

    if audio is not None:
        audio_path = audio

    response = generateOutput(image_path, audio_path, context_text, question)
    if image_path:
        history.append({"role": "user", "content": {"path": image_path}})
    if question:
        history.append({"role": "user", "content": question})
    if audio_path:
        history.append({"role": "user", "content": {"path": audio_path}})

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
)

with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("# Eden")
    gr.Markdown("Chat with the fine-tuned multimodal model using text, audio, or image inputs.")

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        height=250,
        type="messages"
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter your message or upload an image...",
        show_label=False,
        file_types=["image", "audio"],
        container=False,
        scale=3,
        lines=1
    )

    gr.Markdown("Or record a message:")
    audio_input = gr.Audio(type="filepath", sources=["microphone", "upload"])

    chat_input.submit(
        process_input,
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
