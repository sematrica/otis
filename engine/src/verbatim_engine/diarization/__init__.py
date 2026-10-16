"""Speaker Diarization Engine: who spoke when.

The pipeline follows the design of pyannote.audio 3.1's ``SpeakerDiarization``
(segmentation-3.0 + WeSpeaker ResNet34 embeddings + agglomerative clustering),
run on ONNX Runtime so that it needs neither PyTorch nor gated model
downloads:

1. :mod:`.segmentation` finds up to three *local* speakers (and their
   overlaps) in overlapping 10 s windows.
2. :mod:`.embedding` computes a speaker embedding for each local speaker of
   each window.
3. :mod:`.clustering` groups the embeddings into global speakers, estimating
   how many there are unless the user says.
4. :mod:`.pipeline` maps the local speakers onto the global ones and rebuilds
   a frame-level timeline, keeping overlap and per-frame confidence.
5. :mod:`.assign` attaches the timeline to transcript words and segments and
   flags assignments the evidence does not clearly support.

Speakers are anonymous ("Speaker 1", "Speaker 2", ...): nothing here claims
to know who a speaker is.
"""
