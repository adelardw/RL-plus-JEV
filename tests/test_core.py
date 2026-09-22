"""Correctness checks for the pieces whose bugs would be silent."""
from __future__ import annotations

import torch

from rljevf.critic.jev_critic import _interp, fit_calibration, probe_positions
from rljevf.evaluate.gsm8k_eval import extract_answer
from rljevf.evaluate.winrate import _fit_lc
from rljevf.ppo import PPOArgs, PPOTrainer, masked_mean, masked_whiten
from rljevf.rubric import GENERAL_RUBRIC


def test_gae_matches_closed_form():
    """With gamma=lam=1 and a single terminal reward, GAE must reduce to
    R - V(s_t) at every position. If this drifts, PPO silently trains on
    nonsense."""
    T, B = 6, 2
    rewards = torch.zeros(B, T)
    rewards[:, -1] = torch.tensor([2.0, -1.0])
    values = torch.linspace(0.1, 0.6, T).repeat(B, 1)
    mask = torch.ones(B, T)
    lengths = torch.full((B,), T)

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.args = PPOArgs(gamma=1.0, lam=1.0)
    adv, ret = trainer._gae(rewards, values, mask, lengths)

    # V(s_T) = 0 at the terminal state, so A_t = R - V(s_t)
    expected = rewards[:, -1:].expand(B, T) - values
    assert torch.allclose(adv, expected, atol=1e-5), (adv, expected)
    assert torch.allclose(ret, adv + values, atol=1e-5)


def test_gae_discounting():
    """lam<1 must shrink the influence of distant rewards."""
    T = 5
    rewards = torch.zeros(1, T)
    rewards[0, -1] = 1.0
    values = torch.zeros(1, T)
    t1 = PPOTrainer.__new__(PPOTrainer)
    t1.args = PPOArgs(gamma=1.0, lam=1.0)
    t2 = PPOTrainer.__new__(PPOTrainer)
    t2.args = PPOArgs(gamma=1.0, lam=0.5)
    a1, _ = t1._gae(rewards, values, torch.ones(1, T), torch.tensor([T]))
    a2, _ = t2._gae(rewards, values, torch.ones(1, T), torch.tensor([T]))
    assert a1[0, 0] > a2[0, 0]
    assert torch.allclose(a1[0, -1], a2[0, -1])


def test_masked_whiten_ignores_padding():
    x = torch.tensor([[1.0, 2.0, 99.0], [3.0, 4.0, 99.0]])
    m = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    w = masked_whiten(x, m)
    assert abs(float(masked_mean(w, m))) < 1e-5
    real = w[m.bool()]
    assert abs(float(real.std(correction=0)) - 1.0) < 1e-3


def test_calibration_is_exact_on_a_line():
    p = [0.0, 0.25, 0.5, 0.75, 1.0]
    r = [2 * x - 1 for x in p]
    c = fit_calibration(p, r)
    assert abs(c.a - 2.0) < 1e-6 and abs(c.b + 1.0) < 1e-6
    assert abs(c.r2 - 1.0) < 1e-9
    assert abs(c.r2 - c.pearson**2) < 1e-9


def test_calibration_degenerate_input():
    c = fit_calibration([0.5] * 5, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert c.a == 1.0 and c.b == 0.0      # falls back to identity, does not crash


def test_probe_positions_and_interp():
    assert probe_positions(1, 6) == [0]
    assert probe_positions(100, 6)[0] == 0 and probe_positions(100, 6)[-1] == 99
    xs, ys = torch.tensor([0.0, 10.0]), torch.tensor([0.0, 1.0])
    v = _interp(torch.arange(11.0), xs, ys)
    assert abs(float(v[5]) - 0.5) < 1e-5
    assert abs(float(v[0])) < 1e-6 and abs(float(v[10]) - 1.0) < 1e-6


def test_rubric_scalarisation_bounds():
    best = {"correct": 1.0, "helpful": 1.0, "padded": 0.0, "evasive": 0.0, "self_promo": 0.0}
    worst = {"correct": 0.0, "helpful": 0.0, "padded": 1.0, "evasive": 1.0, "self_promo": 1.0}
    assert GENERAL_RUBRIC.scalarise(best) == 2.0
    assert GENERAL_RUBRIC.scalarise(worst) == -3.0
    assert GENERAL_RUBRIC.reward_range == (-3.0, 2.0)

    # `unit` divides by a single constant (the larger of |lo|, |hi|) rather
    # than stretching each side to +/-1: an asymmetric rescale would silently
    # reweight the penalties against the rewards. So the best case maps to
    # 2/3, the worst to -1, and every ratio is preserved.
    assert abs(GENERAL_RUBRIC.scalarise(best, unit=True) - 2 / 3) < 1e-9
    assert GENERAL_RUBRIC.scalarise(worst, unit=True) == -1.0
    mid = {"correct": 0.5, "helpful": 0.5, "padded": 0.25, "evasive": 0.0, "self_promo": 0.0}
    raw, unit = GENERAL_RUBRIC.scalarise(mid), GENERAL_RUBRIC.scalarise(mid, unit=True)
    assert abs(unit - raw / 3.0) < 1e-9


def test_score_normalisation():
    item = [i for i in GENERAL_RUBRIC if i.key == "helpful"][0]
    assert item.n_levels == 3
    assert item.normalise({"score": 2.0}) == 1.0     # top level -> 1.0
    assert item.normalise({"score": 0.0}) == 0.0


def test_length_control_recovers_debiased_rate():
    """A system that wins only because it is longer must get an LC rate near
    0.5 even though its raw rate is high."""
    n = 400
    dl = torch.linspace(-2, 2, n).double()
    w = (dl > 0).double()                  # wins exactly when longer
    theta, gamma = _fit_lc(w, (dl - dl.mean()) / dl.std())
    assert abs(theta) < 0.5, theta
    assert gamma > 1.0, gamma


def test_gsm8k_extraction():
    assert extract_answer("reasoning\n#### 42") == "42"
    assert extract_answer("total is 1,234") == "1234"
    assert extract_answer("#### 18.0") == "18"
    assert extract_answer("no numbers here") == ""
