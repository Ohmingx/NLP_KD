import math
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer

class DistillationTrainer(Seq2SeqTrainer):
    """
    Implements both E2 (standard KD) and E3 (confidence-aware KD).

    L_E2 = alpha * L_CE + (1 - alpha) * L_KD
    L_E3 = alpha * L_CE + (1 - alpha) * sum_t( w_t * L_KD^(t) )

    Set confidence_aware=False for E2, True for E3.
    """

    def __init__(
        self,
        teacher_model=None,
        alpha=0.5,
        temperature=2.0,
        confidence_aware=False,
        confidence_method="entropy",   # "entropy" or "margin"
        weight_power=1.0,              # >1 sharpens weighting, <1 flattens it
        weight_clip=(0.1, 5.0),        # clamp normalized weights for stability
        log_confidence_stats=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        
        # CRITICAL FIX: Move teacher to the GPU before freezing it
        self.teacher = self.teacher.to(self.args.device)
        
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.alpha = alpha
        self.temperature = temperature
        self.confidence_aware = confidence_aware
        self.confidence_method = confidence_method
        self.weight_power = weight_power
        self.weight_clip = weight_clip
        self.log_confidence_stats = log_confidence_stats
        self._step_count = 0

    def _compute_teacher_confidence(self, teacher_logits, mask):
        # Confidence computed from the teacher's TRUE (T=1) distribution
        teacher_probs_true = F.softmax(teacher_logits, dim=-1)

        if self.confidence_method == "entropy":
            eps = 1e-9
            entropy = -(teacher_probs_true * torch.log(teacher_probs_true + eps)).sum(-1, keepdim=True)
            max_entropy = math.log(teacher_logits.size(-1))
            confidence = 1.0 - (entropy / max_entropy)  # higher = more confident
        elif self.confidence_method == "margin":
            top2 = torch.topk(teacher_probs_true, k=2, dim=-1).values
            confidence = (top2[..., 0] - top2[..., 1]).unsqueeze(-1)  # top1 - top2 prob
        else:
            raise ValueError(f"Unknown confidence_method: {self.confidence_method}")

        confidence = confidence.clamp(min=1e-6)
        if self.weight_power != 1.0:
            confidence = confidence.pow(self.weight_power)

        # Normalize so mean weight over valid tokens == 1 
        valid_confidence = confidence[mask]
        mean_conf = valid_confidence.mean().clamp(min=1e-6)
        weights = confidence / mean_conf

        if self.weight_clip is not None:
            weights = weights.clamp(min=self.weight_clip[0], max=self.weight_clip[1])

        return weights, valid_confidence

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]

        student_outputs = model(**inputs)
        student_ce_loss = student_outputs.loss
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )
            teacher_logits = teacher_outputs.logits

        mask = (labels != -100).unsqueeze(-1)  # [batch, seq_len, 1]

        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        kd_loss_per_token = F.kl_div(
            student_log_probs, teacher_probs_soft, reduction="none"
        ).sum(-1, keepdim=True)  # [batch, seq_len, 1]

        if self.confidence_aware:
            weights, valid_confidence = self._compute_teacher_confidence(teacher_logits, mask.squeeze(-1))
            weighted_kd = kd_loss_per_token * weights
            kd_loss = (weighted_kd * mask).sum() / mask.sum()

            if self.log_confidence_stats and self._step_count % max(self.args.logging_steps, 1) == 0:
                self.log({
                    "teacher_confidence_mean": valid_confidence.mean().item(),
                    "teacher_confidence_std": valid_confidence.std().item(),
                    "kd_weight_mean": weights[mask].mean().item(),
                })
        else:
            kd_loss = (kd_loss_per_token * mask).sum() / mask.sum()

        kd_loss = kd_loss * (self.temperature ** 2)
        total_loss = self.alpha * student_ce_loss + (1 - self.alpha) * kd_loss

        self._step_count += 1
        return (total_loss, student_outputs) if return_outputs else total_loss