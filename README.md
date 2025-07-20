REPOSITORY IS A WORK IN PROGRESS

# 3D representations of harmony in Automatic Music Transcription

During this work we will examine the effect of representing frequencies and notes as a musically informed structures within the context of automatic music transcription.
Specifically we will examine the following representations:
- For the notes:
    * Traditional piano-roll notation
    * Helix shaped representations based on the Spiral Note Array model
    * 3D Torus-shaped representations based on Tonnetz, along with multiple structural variations.
- For the spectrum:
    * Traditional 2D spectrogram representations (Log-Mel, CQT)
    * Harmonically enchanced 2D representations (Harmonic CQT)
    * 3D geometries (Torus mapping) 
Each of these representations will be fed into the following network architecture:
(TBD, probably something like a VAE for encoding, a transformer for modeling temporal structure and a generative decoder, insipred by RVQ GAN)

The goal of this work is to analyze the effect of the above representations on the performance of an end-to-end AMT model.

To our knowledge, no prior work has been done on the evaluation of these representation within the context of automatic music transcription.

## Losses

Due to the variations of structure, on top of the commonly used losses in AMT (onset, offset, frame F1), we can employ additional losses both musically and geometrically informed.

# TODO

- [ ] Create a midi parsing method that extracts notes as intervals. These would then be used for converting to other representations.
- [ ] Transfer the SNA representations with modifications to include octaves.