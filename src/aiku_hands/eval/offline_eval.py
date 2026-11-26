"""Offline action-error evaluation of a checkpoint on dataset episodes -> offline_eval.jsonl.

STATUS: not implemented yet (planned after E001 finishes; needs a real checkpoint to validate).
Planned design (to be verified against lerobot 0.7.0 source before use):
- load `runs/<run>/checkpoints/<step>/pretrained_model` via the policy's `from_pretrained`
- rebuild pre/post-processors from the checkpoint (normalization stats travel with it)
- for each held-out episode frame t: predict the full action chunk (no temporal ensembling),
  compare to ground-truth actions t..t+H-1 in *unnormalized* units (-100..100, gripper 0..100)
- record action_l1, action_l1_per_joint[6], chunk_l1_by_horizon[H], gripper_l1
- E001 trained on all 100 episodes -> its numbers are TRAIN-SET numbers; must be labelled split="train".
"""

from __future__ import annotations


def evaluate_checkpoint(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("offline_eval is planned after E001; see module docstring")
