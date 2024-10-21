import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from peft import PeftModel
from .projection import ProjectionBlock

class MultimodalPhiModel(PreTrainedModel):
    def __init__(self, phi_model, tokenizer, projection):
        super().__init__(phi_model.config)
        self.phi_model = phi_model
        self.image_projection = projection
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_name = "microsoft/Phi-3.5-mini-instruct"
        base_phi_model = kwargs.pop('base_phi_model')
        model = PeftModel.from_pretrained(base_phi_model, pretrained_model_name_or_path)
        phi_model = model.merge_and_unload()

        input_dim = 512
        output_dim = 3072

        projector_path = os.path.join(pretrained_model_name_or_path, "image_projector.pth")
        if os.path.exists(projector_path):
            projector_state_dict = torch.load(projector_path, map_location=phi_model.device)
            projector = ProjectionBlock(input_dim, output_dim)
            projector.load_state_dict(projector_state_dict, strict=False)
        else:
            output_dim = phi_model.config.hidden_size
            projector = ProjectionBlock(input_dim, output_dim)

        return cls(phi_model, kwargs.pop('tokenizer'), projector)

    def save_pretrained(self, save_directory):
        self.phi_model.save_pretrained(save_directory)
        projector_path = os.path.join(save_directory, "image_projector.pth")
        torch.save(self.image_projection.state_dict(), projector_path)
        self.config.save_pretrained(save_directory)

    def encode(self, image_features):
        return self.image_projection(image_features)

    def forward(self, start_input_ids, end_input_ids, image_features, attention_mask, labels):
        image_embeddings = self.encode(image_features)
        start_embeds = self.phi_model.get_input_embeddings()(start_input_ids)
        end_embeds = self.phi_model.get_input_embeddings()(end_input_ids)
        input_embeds = torch.cat([start_embeds, image_embeddings, end_embeds], dim=1)

        outputs = self.phi_model(inputs_embeds=input_embeds, 
                                 attention_mask=attention_mask, 
                                 labels=labels, 
                                 return_dict=True)
        
        return outputs

    def gradient_checkpointing_enable(self, **kwargs):
        self.phi_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.phi_model.gradient_checkpointing_disable()