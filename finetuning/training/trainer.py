import os
from typing import Optional
import torch
from transformers import Trainer

class MultimodalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        projection_layer_path = os.path.join(output_dir, "projection_layer")
        os.makedirs(projection_layer_path, exist_ok=True)
        torch.save(self.model.image_projection.state_dict(), os.path.join(projection_layer_path, "pytorch_model.bin"))

        phi_model_path = os.path.join(output_dir, "phi_model")
        os.makedirs(phi_model_path, exist_ok=True)
        self.model.phi_model.save_pretrained(phi_model_path)

        self.model.tokenizer.save_pretrained(output_dir)