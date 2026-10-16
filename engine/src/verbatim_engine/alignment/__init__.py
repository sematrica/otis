"""Word Alignment Engine: precise word timings by CTC forced alignment.

Whisper's own word timestamps come from cross-attention (dynamic time
warping) and are typically off by a tenth of a second or more. Here a
separate CTC acoustic model scores, frame by frame, how likely each of its
tokens is; the known transcript is then forced through those scores with the
Viterbi algorithm (:mod:`.ctc`), which puts every word where the audio
supports it best.

The transcript is never changed: alignment only moves word start/end times.
Words the acoustic model cannot spell (digits, symbols, unsupported scripts)
are absorbed by a wildcard token and timed by it. When the alignment of a
segment is implausible it is discarded, Whisper's timings are kept and the
segment is flagged ``alignment_failed``.
"""
