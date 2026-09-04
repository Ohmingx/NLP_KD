import json

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer


def load_gate_and_pseudolabels(jsonl_path):
    gates = {}
    pseudo_summaries = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            gates[record["example_idx"]] = record["gate"]
            pseudo_summaries[
                record["example_idx"]
            ] = record["pseudo_summary"]

    return gates, pseudo_summaries


class QGHKDTrainer(Seq2SeqTrainer):

    def __init__(
        self,
        teacher_model=None,
        tokenizer_for_pseudo=None,
        example_gates=None,
        pseudo_summaries=None,
        alpha=0.6,
        beta=0.3,
        gamma=0.1,
        temperature=1.5,
        top_k=20,
        max_target_length=256,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.teacher = teacher_model
        self.teacher.eval()

        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

        self.tokenizer_for_pseudo = tokenizer_for_pseudo

        self.example_gates = example_gates or {}
        self.pseudo_summaries = pseudo_summaries or {}

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.temperature = temperature
        self.top_k = top_k
        self.max_target_length = max_target_length

        self._step_count = 0

    def _topk_kd_loss(
        self,
        student_logits,
        teacher_logits,
        mask,
    ):

        teacher_scaled = (
            teacher_logits / self.temperature
        )

        student_scaled = (
            student_logits / self.temperature
        )

        # Select teacher's top-k tokens.
        topk_vals, topk_idx = torch.topk(
            teacher_scaled,
            k=self.top_k,
            dim=-1,
        )

        teacher_topk_probs = F.softmax(
            topk_vals,
            dim=-1,
        )

        # Keep only the student's logits corresponding
        # to the teacher's top-k vocabulary.
        student_topk_logits = torch.gather(
            student_scaled,
            dim=-1,
            index=topk_idx,
        )

        student_topk_logprobs = F.log_softmax(
            student_topk_logits,
            dim=-1,
        )

        kd_per_token = F.kl_div(
            student_topk_logprobs,
            teacher_topk_probs,
            reduction="none",
        ).sum(
            -1,
            keepdim=True,
        )

        kd_loss = (
            kd_per_token * mask
        ).sum() / mask.sum().clamp(min=1)

        return kd_loss * (
            self.temperature ** 2
        )

    def _gates_for_batch(
        self,
        example_indices,
    ):

        gates = [
            self.example_gates.get(
                int(index),
                0.0,
            )
            for index in example_indices
        ]

        return torch.tensor(
            gates,
            dtype=torch.float32,
        )

    def _seq_kd_loss(
        self,
        model,
        inputs,
        example_indices,
        device,
    ):

        pseudo_texts = []
        keep_mask = []

        for index in example_indices:

            index = int(index)

            gate = self.example_gates.get(
                index,
                0.0,
            )

            if (
                gate > 0
                and index in self.pseudo_summaries
            ):
                pseudo_texts.append(
                    self.pseudo_summaries[index]
                )
                keep_mask.append(True)

            else:
                pseudo_texts.append("")
                keep_mask.append(False)

        if not any(keep_mask):
            return torch.tensor(
                0.0,
                device=device,
            )

        pseudo_labels = self.tokenizer_for_pseudo(
            text_target=pseudo_texts,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].to(device)

        pad_id = (
            self.tokenizer_for_pseudo.pad_token_id
        )

        pseudo_labels[
            pseudo_labels == pad_id
        ] = -100

        # Disable the sequence loss for gated-off examples.
        row_mask = torch.tensor(
            keep_mask,
            device=device,
        )

        pseudo_labels[
            ~row_mask
        ] = -100

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=pseudo_labels,
        )

        return outputs.loss

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):

        labels = inputs["labels"]

        example_indices = inputs.pop(
            "example_idx",
            None,
        )

        student_inputs = {
            key: value
            for key, value in inputs.items()
            if key != "example_idx"
        }

        student_outputs = model(
            **student_inputs
        )

        student_ce_loss = (
            student_outputs.loss
        )

        student_logits = (
            student_outputs.logits
        )

        # Teacher is frozen; no gradients here.
        with torch.no_grad():

            teacher_outputs = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )

            teacher_logits = (
                teacher_outputs.logits
            )

        mask = (
            labels != -100
        ).unsqueeze(-1)

        device = student_logits.device

        if example_indices is not None:

            gates = self._gates_for_batch(
                example_indices
            ).to(device)

        else:

            gates = torch.ones(
                student_logits.size(0),
                device=device,
            )

        # Apply one quality gate to every token
        # belonging to the same training example.
        per_example_gate = gates.view(
            -1, 1, 1
        )

        gated_mask = (
            mask
            * per_example_gate
        )

        kd_loss = self._topk_kd_loss(
            student_logits,
            teacher_logits,
            gated_mask,
        )

        seq_kd_loss = torch.tensor(
            0.0,
            device=device,
        )

        if (
            example_indices is not None
            and self.gamma > 0
        ):

            seq_kd_loss = self._seq_kd_loss(
                model,
                inputs,
                example_indices,
                device,
            )

        total_loss = (
            self.alpha * student_ce_loss
            + self.beta * kd_loss
            + self.gamma * seq_kd_loss
        )

        if (
            self._step_count
            % max(self.args.logging_steps, 1)
            == 0
        ):

            self.log(
                {
                    "student_ce_loss":
                        student_ce_loss.item(),

                    "kd_topk_loss":
                        kd_loss.item(),

                    "seq_kd_loss":
                        seq_kd_loss.item(),

                    "batch_gate_mean":
                        gates.mean().item(),
                }
            )

        self._step_count += 1

        if return_outputs:
            return (
                total_loss,
                student_outputs,
            )

        return total_loss