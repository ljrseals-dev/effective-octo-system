# harmonia_neural.py: Whisper backend and bug fixes (revision of the v3 file)

**Status: written by reading; NOT executed.** No torch, transformers or whisper were available where this was written. Syntax, undefined
names and shape arithmetic were checked; `test_harmonia_neural_fixes.py` (16 tests, no downloads) was never run. Run it with torch installed
before trusting any item here, and confirm each test also fails on `harmonia_neural_v3_original` (change the import line).

Base: `harmonia_neural__3_.py` (1284 lines), the newest of the four uploaded copies (it alone has the streaming CausalConv1d phase fix).
The other three are superseded and can be deleted. `train.py` (uploaded earlier) belongs to a different project (neural compressor) and was not touched.

| # | Change | Why | Evidence |
|---|--------|-----|----------|
| 1 | Whisper semantic backend (`ModelCfg.semantic_backend`, default `"whisper"`; also `wavlm`, `mfcc`, `auto`) | requested | not run |
| 2 | `harp_loss_stft` squeezes `(B,1,T)` to `(B,T)`, Hann window, real bin count | `torch.stft` takes 1-D/2-D input; README Known issue #2 | not run |
| 3 | `train()` refuses a silent MFCC fallback (preflight + post-build check; `--allow-semantic-fallback` to override) | README Known issue #3 | not run |
| 4 | Discriminator loss built outside `torch.no_grad()` (`_d_step`) | v3 called `.backward()` on a loss computed under `no_grad`, which raises | the premise is tested in the test file; not run |
| 5 | Diffusion head resamples semantic frames to the latent frame count | latents ~120 fps, Whisper/WavLM ~50, MFCC ~100; they could not be added | shape arithmetic (1 s: 120 / 50 / 49 / 98 frames) |
| 6 | `_to_16k`: linear-interp fallback instead of feeding 24 kHz audio to a 16 kHz model | silent wrong rate without torchaudio | not run |
| 7 | `SemanticAnchor.train()` keeps frozen backbones in eval | `model.train()` re-enabled WavLM dropout | not run |
| 8 | `load_ckpt` also compares `use_semantic`, `semantic_backend`, `whisper_model` | clear error instead of a state_dict mismatch | not run |

Whisper details: uses the **openai-whisper** package (`pip install openai-whisper`; the PyPI package named plain `whisper` is a different project). The
encoder is run by hand (conv1, conv2, blocks, ln_post) with the positional embedding sliced, because the stock forward asserts a 30 s input. Whisper's
log-mel recipe is reproduced (n_fft 400, hop 160, floor at max-8, (x+4)/4) and the filterbank comes from `whisper.audio.mel_filters`. Clips over 30 s raise.
`whisper_layer=-1` is the final (post-LN) output, `0..n_blocks` the state after that many blocks. First use of `whisper.load_model` downloads weights (cached under
`~/.cache/whisper`, per that package); this is a new network touch point to add to the README's network section. Default backend changes from WavLM to
Whisper, so any checkpoint trained with the old default will not load (none ship).

Not touched: `CausalTransposedConv1d` (known broken), VRVQ architecture, numpy codec. One observation, not a bug fix: v3's `CausalConv1d` runs the conv at stride 1 and
subsamples, which costs `stride` times the compute for the encoder's strided layers.

Run: `pip install torch torchaudio numpy scipy soundfile pytest` then `pytest -q test_harmonia_neural_fixes.py`.
