# src/vrpo_trainer.py

from transformers import Trainer
from .vrpo_elbo import vrpo_loss_on_batch


class DIFFA_VRPOTrainer(Trainer):
    def __init__(self,
                 ref_model,
                 beta: float = 0.1,
                 mc_steps: int = 8,
                 mask_id: int = 126336,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.mc_steps = mc_steps
        self.mask_id = mask_id
        
        device = self.args.device
        self.ref_model.to(device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        loss = vrpo_loss_on_batch(
            model_theta=model,
            model_ref=self.ref_model,
            batch=inputs,
            beta=self.beta,
            mc_steps=self.mc_steps,
            mask_id=self.mask_id,
        )
        return (loss, None) if return_outputs else loss
