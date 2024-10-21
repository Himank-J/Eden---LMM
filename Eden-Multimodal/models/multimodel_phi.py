import torch
import torch.nn as nn
import os
from peft import PeftModel
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from .projection_layer import ProjectionBlock

class MultimodalPhiModel(PreTrainedModel):
    def gradient_checkpointing_enable(self, **kwargs):
        self.phi_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.phi_model.gradient_checkpointing_disable()

    def __init__(self, phi_model, tokenizer, projection):
        super().__init__(phi_model.config)
        self.phi_model = phi_model
        self.image_projection = projection
        self.tokenizer = tokenizer
        self.base_phi_model = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, debug=False, **kwargs):
        model_name = "microsoft/Phi-3.5-mini-instruct"
        base_phi_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        phi_path = pretrained_model_name_or_path

        model = PeftModel.from_pretrained(base_phi_model, phi_path)
        phi_model = model.merge_and_unload()

        input_dim = 512
        output_dim = 3072

        projector_path = os.path.join(pretrained_model_name_or_path, "image_projector.pth")
        if os.path.exists(projector_path):
            projector_state_dict = torch.load(projector_path, map_location=phi_model.device)
            projector = ProjectionBlock(input_dim, output_dim)
            projector.load_state_dict(projector_state_dict, strict=False)
            print(f"Loaded projector with input_dim={input_dim}, output_dim={output_dim}")
        else:
            print(f"Projector weights not found at {projector_path}. Initializing with default dimensions.")
            input_dim = 512
            output_dim = phi_model.config.hidden_size
            projector = ProjectionBlock(input_dim, output_dim)

        model = cls(phi_model, tokenizer, projector)
        model.base_phi_model = base_phi_model
        return model

    def save_pretrained(self, save_directory):
        self.phi_model.save_pretrained(save_directory)
        projector_path = os.path.join(save_directory, "image_projector.pth")
        torch.save(self.image_projection.state_dict(), projector_path)
        self.config.save_pretrained(save_directory)

    def encode(self, image_features):
        image_projections = self.image_projection(image_features)
        return image_projections

    def forward(self, start_input_ids, end_input_ids, image_features, attention_mask, labels):
        device = next(self.parameters()).device

        start_embeds = self.phi_model.get_input_embeddings()(start_input_ids.to(device))
        end_embeds = self.phi_model.get_input_embeddings()(end_input_ids.to(device))

        if image_features is not None:
            image_embeddings = self.encode(image_features.to(device)).bfloat16()
            input_embeds = torch.cat([start_embeds, image_embeddings, end_embeds], dim=1)
        else:
            input_embeds = torch.cat([start_embeds, end_embeds], dim=1)

        outputs = self.phi_model(inputs_embeds=input_embeds.to(device), 
                                 attention_mask=attention_mask.to(device), 
                                 labels=labels, 
                                 return_dict=True)
        
        return outputs