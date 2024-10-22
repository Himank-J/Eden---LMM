# Eden: Small Multi-Modal 

## 🎯 Objective: 


The project aimed to finetune a multi-modal language model that accepts text, image, and audio inputs and generates text output based on a query.

---

## 📑 Dataset:

- Instruct 150k: A carefully curated dataset designed to enhance the model’s instruction-following capabilities.
- COCO Images: Original images from the COCO dataset were paired with text data to create a rich, multi-modal learning environment.

---

## :basecamp: Model Architecture:

- Llava Model Approach: The project adopted a Llava-like framework, leveraging the CLIP model for image embeddings. These embeddings were fine-tuned for textual alignment.

- CLIP Pretrained Model:

  - Extracts image embeddings which are then adjusted to align with text representations.

- Projection Layer:

  - A transformation layer that adjusts CLIP image embeddings into a form that is compatible with the downstream language model, ensuring seamless integration with Phi3.5.

- Phi3.5_mini_instruct Model:

  - Acts as the core language model (SLM).
  - Fine-tuned using QLora for efficient instruction-tuning and to comprehend image embeddings effectively.

- Whisper for Audio:

  - Whisper model converts audio inputs into text, which is combined with other textual data and questions.
  - Both the audio-converted text and original textual inputs are passed to the Phi3.5 layer for comprehensive processing.

--- 

## 📍 Deployment:

- The final model was deployed on Huggingface, allowing users to submit multi-modal inputs (image, audio, and/or text) along with a query.
- The model generates a coherent text-based output, informed by all available input modalities and the given question.

---

## ⭐ Significance: 
This model demonstrates a flexible and scalable approach to multi-modal AI by integrating advanced models like CLIP, Phi3.5, and Whisper, enabling real-world applications such as conversational AI, visual Q&A, and multimedia interpretation.

---

## Core Components

### 1. Models and Processors

- CLIP: Used for image feature extraction
  - Model: openai/clip-vit-base-patch32
  - Processor: Handles image preprocessing

- Phi-3.5: Base language model
  - Model: microsoft/Phi-3.5-mini-instruct
  - Configured with 4-bit quantization for memory efficiency
  - Uses bfloat16 for computation

### 2. Data Collation

- Batching of image features
- Tokenization of text segments
- Creation of attention masks
- Label generation for causal language modeling
- Special handling of answer sections for loss calculation

### 3. Projection Block

- Projects CLIP image features into Phi's embedding space:
```
CopyInput (512) → LayerNorm → Linear → GELU → Linear → Output (3072)
```

- MultimodalPhiModel
  - Core multimodal architecture that:
  - Inherits from PreTrainedModel
  - Combines projected image features with text embeddings
  - Handles model saving/loading
  - Processes both start and end text segments

### 4. Training Implementation

  - Model Configuration
    - Uses LoRA for efficient fine-tuning
    - Parameters:
      - lora_alpha: 16
      - lora_dropout: 0.1
      - r: 64
       
    - Targets specific modules:
      - o_proj
      - qkv_proj
      - gate_up_proj

  - Training Configuration
    - Batch size: 2 (per device)
    - Maximum steps: 6000
    - Save frequency: Every 40% of training
    - Logging frequency: Every 10% of training
    - Evaluation frequency: Every 10% of training
    - Uses bfloat16 precision
    - Implements gradient checkpointing

### 5. Memory and Performance Optimizations

  - 4-bit quantization for the base model
  - Gradient checkpointing enabled
  - LoRA for efficient fine-tuning
  - bfloat16 precision training
  - Selective parameter updates

---

## Data Flow

1. Image → CLIP → Image Features (512 dim)
2. Image Features → ProjectionBlock → Projected Features (3072 dim)
3. Text → Tokenizer → Token Embeddings
4. Concatenation: [Start Text Embeddings + Projected Image Features + End Text Embeddings]
5. Combined Input → Phi-3.5 → Generated Response
