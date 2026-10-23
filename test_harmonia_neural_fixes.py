"""Tests for the harmonia_neural.py revision (Whisper backend + bug fixes).

STATUS: WRITTEN BUT NOT RUN. The environment these were written in had no torch.
Run them yourself:   pip install torch torchaudio numpy scipy soundfile pytest
                     pytest -q test_harmonia_neural_fixes.py
None of these tests download anything: Whisper is replaced by a stub encoder and
a stub mel filterbank, and the checks are about plumbing and shapes, not about
how good Whisper's features are.

Each test notes how it behaves on the ORIGINAL v3 file (by reading it): the
repo's own rule is that a test must fail on the buggy code and pass on the fixed
code, so please confirm that direction too (run against harmonia_neural_v3_original.py
by changing the import line).
"""
import json
import types

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import harmonia_neural as hn


class StubWhisperEncoder(nn.Module):
    """Same attribute surface as openai-whisper's AudioEncoder (conv1, conv2,
    positional_embedding, blocks, ln_post), tiny and randomly initialised."""

    def __init__(self, n_mels=80, d=32, n_blocks=3, n_ctx=1500):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.register_buffer("positional_embedding", torch.randn(n_ctx, d))
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, d), nn.GELU()) for _ in range(n_blocks)])
        self.ln_post = nn.LayerNorm(d)


@pytest.fixture
def fake_mel(monkeypatch):
    monkeypatch.setattr(
        hn.SemanticAnchor, "_mel_filters",
        staticmethod(lambda device, n_mels: torch.ones(n_mels, 201, device=device) / 201))


# ---------------------------------------------------------------- Whisper backend
def test_whisper_anchor_shapes_with_injected_encoder(fake_mel):
    enc = StubWhisperEncoder()
    sem = hn.SemanticAnchor(sr=24000, backend="whisper", whisper_encoder=enc, proj_dim=48)
    assert sem.backend_active == "whisper" and not sem.using_fallback
    f, valid = sem(torch.randn(2, 24000))
    # 24000 samples -> 16000 at 16 kHz -> 100 mel frames -> conv2 stride 2 -> 50 frames
    assert f.shape == (2, 50, 48) and valid == 32
    assert not any(p.requires_grad for p in enc.parameters()), "backbone must be frozen"


def test_whisper_layer_selection_and_validation(fake_mel):
    x = torch.randn(1, 24000)
    enc = StubWhisperEncoder()                  # ONE encoder, so differences come from the layer choice only
    outs = {}
    for k in (-1, 0, 1, 3):
        sem = hn.SemanticAnchor(backend="whisper", whisper_encoder=enc,
                                whisper_layer=k, proj_dim=32)
        sem.projector = nn.Identity()           # compare raw backbone features
        outs[k] = sem._features(x)[0]
    assert all(o.shape == (1, 50, 32) for o in outs.values())
    assert not torch.allclose(outs[0], outs[1]) and not torch.allclose(outs[1], outs[3])
    assert not torch.allclose(outs[3], outs[-1]), "-1 applies ln_post, 3 stops before it"
    sem = hn.SemanticAnchor(backend="whisper", whisper_encoder=enc, whisper_layer=9, proj_dim=32)
    with pytest.raises(ValueError):
        sem(x)


def test_whisper_rejects_audio_longer_than_the_encoder_context(fake_mel):
    sem = hn.SemanticAnchor(backend="whisper", whisper_encoder=StubWhisperEncoder(n_ctx=60),
                            proj_dim=32)
    with pytest.raises(ValueError, match="too long"):
        sem(torch.randn(1, 48000))              # 2 s -> 100 frames > 60


def test_whisper_encoder_missing_attribute_is_reported_not_swallowed(fake_mel):
    sem = hn.SemanticAnchor(backend="whisper", whisper_encoder=nn.Linear(2, 2))
    assert sem.using_fallback and sem.backend_active == "mfcc"
    assert "'conv1'" in sem.fallback_reason


def test_whisper_unavailable_falls_back_and_says_why(monkeypatch):
    monkeypatch.setattr(hn, "WHISPER_AVAILABLE", False)
    sem = hn.SemanticAnchor(backend="whisper")
    assert sem.using_fallback and sem.backend_active == "mfcc"
    assert "openai-whisper" in sem.fallback_reason
    f, _ = sem(torch.randn(1, 24000))           # MFCC path still works
    assert f.shape[0] == 1 and f.shape[-1] == 768


def test_frozen_backbone_stays_in_eval_after_model_train(fake_mel):
    sem = hn.SemanticAnchor(backend="whisper", whisper_encoder=StubWhisperEncoder())
    sem.train()
    assert not sem.whisper_enc.training         # v3: train() flipped the frozen backbone into train mode


def test_resample_to_16k_never_returns_the_wrong_rate(monkeypatch):
    x = torch.randn(2, 24000)
    assert hn._to_16k(x, 16000) is x
    monkeypatch.setattr(hn, "HAVE_TA", False)   # v3: audio went to WavLM at 24 kHz, silently
    assert hn._to_16k(x, 24000).shape == (2, 16000)


# ---------------------------------------------------------------- frame-rate alignment
def test_align_time():
    x = torch.randn(2, 50, 7)
    assert hn._align_time(x, 50) is x
    y = hn._align_time(x, 120)
    assert y.shape == (2, 120, 7)


def test_diffusion_head_accepts_semantic_frames_at_a_different_rate():
    # v3: lat_in(z) (B,120,h) + sem_proj(sem) (B,50,h) cannot broadcast -> RuntimeError
    dh = hn.DiffusionHeadNN(latent_dim=8, sem_dim=12)
    z1, sem = torch.randn(2, 120, 8), torch.randn(2, 50, 12)
    loss = dh.loss(z1, sem)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert dh.sample(sem, 120, 8, steps=2).shape == (2, 120, 8)


# ---------------------------------------------------------------- HARP loss shape
class _StubHarp:
    def __init__(self):
        self.calls = []

    def band_weight(self, stage, n_bins, device):
        self.calls.append((stage, n_bins))
        return torch.ones(n_bins, device=device)


def test_harp_loss_stft_accepts_the_real_call_site_shape():
    # _one_step passes xw (B, 1, T). v3: torch.stft on 3-D input errors / mis-sizes the bin axis.
    harp = _StubHarp()
    x = torch.randn(2, 1, 8000)
    stages = [torch.randn(2, 1, 8000), torch.randn(2, 1, 8000)]
    loss = hn.harp_loss_stft(stages, x, harp, torch.device("cpu"))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(n == 513 for _, n in harp.calls), "band_weight must receive the STFT bin count"
    assert torch.isfinite(hn.harp_loss_stft(stages, x.squeeze(1), harp, torch.device("cpu")))
    with pytest.raises(ValueError):
        hn.harp_loss_stft(stages, torch.randn(2, 2, 8000), harp, torch.device("cpu"))
    assert hn.harp_loss_stft([], x, harp, torch.device("cpu")).item() == 0.0


# ---------------------------------------------------------------- discriminator step
def test_premise_loss_built_under_no_grad_cannot_backward():
    lin = nn.Linear(1, 1)
    with torch.no_grad():
        loss = lin(torch.ones(1, 1)).sum()
    with pytest.raises(RuntimeError):
        loss.backward()                          # this is what v3's inline D step did


def test_d_step_builds_a_graph_and_updates_the_discriminators():
    cfg = hn.TrainCfg(device="cpu", grad_clip=1.0)
    mcfg = hn.ModelCfg(sr=24000, n_stages=2, n_experts=2, top_k=1, latent_dim=16,
                       enc_dim=32, codebook_dim=16, use_diffusion=False,
                       use_semantic=False, use_otcfm=False)
    model, msd, mpd = hn.NeuralHarmonia(mcfg), hn.MSD(), hn.MPD()
    opt_d = torch.optim.AdamW(list(msd.parameters()) + list(mpd.parameters()), lr=1e-3)
    before = [p.detach().clone() for p in msd.parameters()]
    d_loss = hn._d_step(model, msd, mpd, cfg, torch.randn(2, 24000) * 0.1, opt_d)
    assert torch.isfinite(d_loss)
    assert any(not torch.equal(b, p) for b, p in zip(before, msd.parameters()))


# ---------------------------------------------------------------- train() guards
def test_train_refuses_silent_semantic_fallback(monkeypatch):
    monkeypatch.setattr(hn, "WHISPER_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="refusing silent MFCC fallback"):
        hn.train(hn.TrainCfg(device="cpu"),
                 hn.ModelCfg(use_semantic=True, semantic_backend="whisper"), files=[])


def test_semantic_preflight_cases(monkeypatch):
    monkeypatch.setattr(hn, "WHISPER_AVAILABLE", False)
    monkeypatch.setattr(hn, "WAVLM_AVAILABLE", False)
    hn.semantic_preflight(hn.ModelCfg(use_semantic=False))                              # off: fine
    hn.semantic_preflight(hn.ModelCfg(use_semantic=True, semantic_backend="mfcc"))      # deliberate MFCC: fine
    hn.semantic_preflight(hn.ModelCfg(use_semantic=True, semantic_backend="whisper"),
                          allow_fallback=True)                                          # explicit opt-in: fine
    for b in ("whisper", "wavlm", "auto"):
        with pytest.raises(RuntimeError):
            hn.semantic_preflight(hn.ModelCfg(use_semantic=True, semantic_backend=b))
    with pytest.raises(ValueError):
        hn.semantic_preflight(hn.ModelCfg(use_semantic=True, semantic_backend="nope"))


def test_semantic_postcheck_catches_weights_that_failed_to_load():
    sem = types.SimpleNamespace(using_fallback=True, backend_requested="whisper",
                                fallback_reason="whisper: URLError: offline")
    model = types.SimpleNamespace(sem=sem)
    with pytest.raises(RuntimeError, match="failed to load"):
        hn.semantic_postcheck(model)
    hn.semantic_postcheck(model, allow_fallback=True)
    hn.semantic_postcheck(types.SimpleNamespace(sem=None))


# ---------------------------------------------------------------- config compatibility
def test_old_config_files_without_the_new_fields_still_load(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"sr": 24000, "n_stages": 9, "use_semantic": True}))
    cfg = hn.load_config(str(p))
    assert cfg.semantic_backend == "whisper" and cfg.whisper_model == "base" and cfg.whisper_layer == -1
