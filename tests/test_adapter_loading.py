"""A LoRA checkpoint must not load as its own base model.

This is the failure that motivated the test. `AutoModelForCausalLM.from_pretrained`
on an adapter-only directory returns the base model with the adapter silently
ignored: no error, no warning, weights bit-identical to the untrained
checkpoint. Every arm would then evaluate as its own baseline, every comparison
would come out flat, and the study would conclude that the choice of reward
source does not matter -- which is exactly the kind of null result that looks
like a finding.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from rljevf.evaluate.generate import load_policy

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _trained_adapter(tmp_path):
    """A checkpoint whose adapter is demonstrably non-zero."""
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    base.save_pretrained(tmp_path / "base")
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(MODEL).save_pretrained(tmp_path / "base")

    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(tmp_path / "base", dtype=torch.float32),
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                   task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]),
    )
    # lora_B initialises to zero, so an untouched adapter is a no-op and would
    # pass this test vacuously; give it real values
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.02)
    model.save_pretrained(tmp_path / "ckpt")
    return tmp_path / "base", tmp_path / "ckpt"


def test_adapter_checkpoint_differs_from_its_base(tmp_path):
    base_dir, ckpt_dir = _trained_adapter(tmp_path)

    base, _, _ = load_policy(str(base_dir), device="cpu", dtype=torch.float32)
    ckpt, _, _ = load_policy(str(ckpt_dir), device="cpu", dtype=torch.float32)

    bd, cd = base.state_dict(), ckpt.state_dict()
    shared = [k for k in bd if k in cd]
    assert shared, "the two checkpoints share no parameters -- loading is broken"
    differing = [k for k in shared if not torch.allclose(bd[k], cd[k], atol=1e-7)]
    assert differing, (
        "the adapter checkpoint loaded bit-identical to its base: the adapter "
        "was dropped, and every evaluation of it would measure the untrained model"
    )
    # the adapter targets q_proj and v_proj, so the difference must land there
    assert any("q_proj" in k or "v_proj" in k for k in differing)


def test_adapter_without_its_base_fails_loudly(tmp_path):
    """An adapter is useless without the model it adapts; say so rather than
    quietly returning something."""
    import json

    _, ckpt_dir = _trained_adapter(tmp_path)
    cfg = ckpt_dir / "adapter_config.json"
    d = json.loads(cfg.read_text())
    d["base_model_name_or_path"] = str(tmp_path / "gone")
    cfg.write_text(json.dumps(d))

    try:
        load_policy(str(ckpt_dir), device="cpu", dtype=torch.float32)
    except (FileNotFoundError, OSError, ValueError):
        return
    raise AssertionError("loading an adapter with no base model must fail loudly")
