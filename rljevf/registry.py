"""One place that turns a RunConfig into the objects a trainer needs.

Keeping construction here is what lets train_grpo.py and train_ppo.py stay
identical across arms: the run differs by config, not by code path.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from .config import API_JUDGE_MODEL, BERT_RM_MODEL, JUDGE_MODEL, RunConfig
from .critic.jev_critic import JevCritic, LearnedValueCritic
from .jevclient import JevClient, Meter
from .orclient import ChatClient
from .rewards.base import BaseRewardSource
from .rubric import GENERAL_RUBRIC
from .sg2jev import (
    GeneratedRubricSource,
    PolicyProposedRubricSource,
    StaticRubricSource,
)


def build_rubric_source(cfg: RunConfig, policy_generate_fn=None):
    if cfg.rubric_source == "static":
        return StaticRubricSource(GENERAL_RUBRIC)
    if cfg.rubric_source == "generated":
        return GeneratedRubricSource(
            ChatClient("openai/gpt-4.1-mini", meter=Meter(name="sg2jev_gen"))
        )
    if cfg.rubric_source == "policy":
        if policy_generate_fn is None:
            raise ValueError("rubric_source='policy' needs a generate_fn from the policy")
        return PolicyProposedRubricSource(policy_generate_fn)
    raise ValueError(cfg.rubric_source)


def build_reward(
    cfg: RunConfig,
    policy=None,
    tokenizer=None,
    policy_generate_fn=None,
) -> tuple[Callable[..., Any], BaseRewardSource | None]:
    """-> (callable with TRL's signature, the underlying source for .stats())."""
    src_name = cfg.reward_source

    if src_name == "none":
        def zero_reward(prompts, completions, **kw):
            return [0.0] * len(prompts)
        return zero_reward, None

    if src_name == "jev":
        from .rewards.jev_rf import JevRewardSource

        source = JevRewardSource(
            client=JevClient(meter=Meter(name=f"jev_{cfg.run_id}")),
            rubric_source=build_rubric_source(cfg, policy_generate_fn),
            unit_scale=(cfg.rubric_source != "static"),
        )
        # TRL detects async reward functions with inspect.iscoroutinefunction,
        # which is False for an instance with an async __call__. Wrap it in a
        # real coroutine function so the trainer runs it on its async loop.
        async def jev_reward(prompts, completions, **kw):
            return await source(prompts, completions, **kw)

        jev_reward.__name__ = "jev_reward"
        return jev_reward, source

    if src_name == "api_judge":
        from .rewards.api_judge import APIJudgeRewardSource

        source = APIJudgeRewardSource(API_JUDGE_MODEL, name=f"apijudge_{cfg.run_id}")

        async def api_judge_reward(prompts, completions, **kw):
            return await source(prompts, completions, **kw)

        api_judge_reward.__name__ = "api_judge_reward"
        return api_judge_reward, source

    if src_name == "bert_rm":
        from .rewards.bert_rm import BertRMRewardSource

        source = BertRMRewardSource(cfg.bert_rm_model if hasattr(cfg, "bert_rm_model") else BERT_RM_MODEL)

        def bert_reward(prompts, completions, **kw):
            return source(prompts, completions, **kw)

        bert_reward.__name__ = "bert_rm_reward"
        return bert_reward, source

    if src_name in ("llm_judge", "self_judge"):
        from .rewards.llm_judge import LLMJudgeRewardSource, SelfJudgeRewardSource

        if src_name == "self_judge":
            # a *frozen copy* of the SFT checkpoint, not the live policy
            source = SelfJudgeRewardSource(
                model_name=cfg.sft_checkpoint or cfg.policy_model, rubric=GENERAL_RUBRIC
            )
        else:
            source = LLMJudgeRewardSource(model_name=JUDGE_MODEL, rubric=GENERAL_RUBRIC)

        def judge_reward(prompts, completions, **kw):
            return source(prompts, completions, **kw)

        judge_reward.__name__ = f"{src_name}_reward"
        return judge_reward, source

    if src_name == "self_certainty":
        from .rewards.self_certainty import SelfCertaintyRewardSource

        source = SelfCertaintyRewardSource(model=policy, tokenizer=tokenizer)

        def certainty_reward(prompts, completions, **kw):
            return source(prompts, completions, **kw)

        certainty_reward.__name__ = "self_certainty_reward"
        return certainty_reward, source

    raise ValueError(f"unknown reward source {src_name!r}")


def build_critic(cfg: RunConfig, policy=None, tokenizer=None):
    if cfg.critic == "jev":
        return JevCritic(
            tokenizer=tokenizer,
            client=JevClient(meter=Meter(name=f"jevcritic_{cfg.run_id}")),
            num_prefixes=cfg.jev_critic_num_prefixes,
        )
    if cfg.critic == "learned":
        return LearnedValueCritic(policy)
    raise ValueError(f"PPO needs critic in {{jev, learned}}, got {cfg.critic!r}")
