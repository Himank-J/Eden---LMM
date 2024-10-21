import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from models.multimodal_phi import MultimodalPhiModel
from models.projection import ProjectionBlock
from data.dataset import ImageTextDatasetForCausalLM
from training.trainer import MultimodalTrainer
from utils.logging_utils import logger
from config import MODEL_NAME, CLIP_MODEL_NAME, BNB_CONFIG, LORA_CONFIG, TRAINING_ARGS, MAX_LENGTH

from transformers import CLIPProcessor, CLIPModel

def collate_fn(batch):
    image_features = torch.stack([item['image_features'] for item in batch])
    start_texts = [item['start_text'] for item in batch]
    end_texts = [item['end_text'] for item in batch]

    batch_size = image_features.shape[0]
    num_image_tokens = image_features.shape[1]
    
    image_tokens = torch.full((batch_size, num_image_tokens), -100, dtype=torch.long)

    start_tokens = tokenizer(start_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    end_tokens = tokenizer(end_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    
    start_input_ids = start_tokens['input_ids']
    start_attention_mask = start_tokens['attention_mask']
    end_input_ids = end_tokens['input_ids']
    end_attention_mask = end_tokens['attention_mask']

    input_ids = torch.cat([start_input_ids, image_tokens, end_input_ids], dim=1)
    attention_mask = torch.cat([start_attention_mask, torch.ones((batch_size, num_image_tokens), dtype=torch.long), end_attention_mask], dim=1)

    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100

    answer_start = (input_ids == tokenizer.encode("<|assistant|>", add_special_tokens=False)[0]).nonzero(as_tuple=True)[1]
    for i, start in enumerate(answer_start):
        labels[i, :start] = -100

    return {
        "start_input_ids": start_input_ids,
        "end_input_ids": end_input_ids,
        "image_features": image_features,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    clipmodel = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clipprocessor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    phi_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        quantization_config=BNB_CONFIG,
        trust_remote_code=True,
    )
    phi_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = ImageTextDatasetForCausalLM(clipprocessor, "./conversations.csv", "./data/train2014", tokenizer)
    dataset.shuffle(seed=42)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

    input_dim = 512
    output_dim = 3072
    projector = ProjectionBlock(input_dim, output_dim)

    model = MultimodalPhiModel(phi_model, tokenizer, projector)

    peft_config = LoraConfig(**LORA_CONFIG)
    model.phi_model = get_peft_model(model.phi_model, peft_config)
    model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params

    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"All parameters: {all_params:,}")
    logger.info(f"Percentage of trainable parameters: {trainable_percent:.2f}%")

    trainer = MultimodalTrainer(
        model=model,
        args=TrainingArguments(**TRAINING_ARGS),
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    logger.info("Saving model...")
    model.save_pretrained("./checkpoints")
    logger.info("Model saved successfully.")