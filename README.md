# Project Contributors:

Ishan Yadav (iy2159)  
Riyam Patel (rp4334)  
Nived Damodaran (nd2746)

# Language-Based Audio Retrieval

This repository contains the implementation of our work on Language-Based Audio Retrieval (LBAR) systems, which retrieve audio signals based on their textual descriptions (audio captions).

## Overview

Language-Based Audio Retrieval involves matching text queries with relevant audio files. For each text query, our system retrieves and ranks audio files based on their relevance to the query. Our approach builds upon and enhances existing baseline and state-of-the-art methods.

## Features

- **Multiple Audio Encoders**: Implementation of various architectures including:
  - CRNN (Convolutional Recurrent Neural Network)
  - TCN (Temporal Convolutional Network)
  - Spec2Vec Encoder
  - Custom CNN + BEATs

- **Multiscale Mel Spectrograms**: Enhanced audio representation that captures both long and short audio events effectively.

- **Multiple Loss Functions**:
  - InfoNCE Loss
  - Cosine Loss
  - VicReg Loss

## Model Architecture

Our system follows a bi-encoder architecture with:
- Separate encoders for audio and text inputs
- Audio branch using various encoder architectures
- Text branch using Sentence-BERT or RoBERTa encoders
- Match mechanism computing similarity between audio and text embeddings

### Audio Encoders

#### CRNN Encoder
- CNN captures local time-frequency patterns
- GRU captures sequential information
- Bidirectionality allows future and past context modeling

#### TCN Encoder
- Captures long-term temporal dependencies without recurrence
- Parallelizable and faster to train than RNNs
- Robust to vanishing gradients and more stable on longer sequences

#### Spec2Vec Encoder
- Simple and fast to train
- Learns holistic representations of entire spectrograms
- Works well when input dimensions are consistent and fixed

#### Custom CNN + BEATs Encoder
- 3D CNN captures both spectral and short-term temporal correlations
- BEATs was pretrained in a setup aligned with language for better audio-text matching

## Dataset

We use the Clotho v2 dataset for training and evaluation, which includes:
- 6,974 audio samples (15-30 seconds each) from Freesound
- 34,870 captions (8-20 words each)
- Development-training set: 3,839 audio clips with 19,195 captions
- Development-validation set: 1,045 clips with 5,225 captions
- Development-testing set: 1,045 clips with 5,225 captions

## Evaluation Metrics

Our system is evaluated using standard retrieval metrics:
- **R@1**: Recall score for the top-1 retrieved result
- **R@5**: Recall among the top-5 retrieved results
- **R@10**: Recall among the top-10 retrieved results
- **mAP@10**: Mean Average Precision across the top-10 results

## Usage

Run the `main.ipynb` file to train and evaluate models.  
You can plug in any model from the `models/` directory by updating the configuration accordingly.

---

## Results

| Model          | Spectrogram     | Loss     | R1    | R5    | R10   | mAP10 |
|----------------|------------------|----------|-------|-------|-------|--------|
| Baseline       | Normal Mel       | Standard | 0.130 | 0.343 | 0.480 | 0.222  |
| CRNN           | Normal Mel       | InfoNCE  | 0.085 | 0.285 | 0.350 | 0.108  |
|                | Normal Mel       | VICReg   | 0.079 | 0.218 | 0.340 | 0.102  |
|                | Normal Mel       | Cosine   | 0.072 | 0.210 | 0.325 | 0.102  |
|                | Multiscale Mel   | InfoNCE  | 0.045 | 0.135 | 0.235 | 0.059  |
|                | Multiscale Mel   | VICReg   | 0.039 | 0.125 | 0.215 | 0.053  |
|                | Multiscale Mel   | Cosine   | 0.032 | 0.115 | 0.195 | 0.046  |
| TCN            | Normal Mel       | InfoNCE  | 0.080 | 0.225 | 0.345 | 0.109  |
|                | Normal Mel       | VICReg   | 0.073 | 0.215 | 0.325 | 0.097  |
|                | Normal Mel       | Cosine   | 0.069 | 0.200 | 0.312 | 0.094  |
|                | Multiscale Mel   | InfoNCE  | 0.042 | 0.128 | 0.220 | 0.058  |
|                | Multiscale Mel   | VICReg   | 0.036 | 0.120 | 0.200 | 0.055  |
|                | Multiscale Mel   | Cosine   | 0.030 | 0.105 | 0.185 | 0.042  |
| Spec2Vec       | Normal Mel       | InfoNCE  | 0.075 | 0.215 | 0.330 | 0.103  |
|                | Normal Mel       | VICReg   | 0.068 | 0.205 | 0.310 | 0.100  |
|                | Normal Mel       | Cosine   | 0.063 | 0.195 | 0.300 | 0.085  |
|                | Multiscale Mel   | InfoNCE  | 0.038 | 0.122 | 0.210 | 0.055  |
|                | Multiscale Mel   | VICReg   | 0.032 | 0.112 | 0.195 | 0.049  |
|                | Multiscale Mel   | Cosine   | 0.028 | 0.098 | 0.175 | 0.043  |
| CNN + BEATS    | Normal Mel       | InfoNCE  | 0.082 | 0.225 | 0.345 | 0.093  |
|                | Normal Mel       | VICReg   | 0.075 | 0.215 | 0.335 | 0.090  |
|                | Normal Mel       | Cosine   | 0.070 | 0.205 | 0.325 | 0.083  |
|                | Multiscale Mel   | InfoNCE  | 0.040 | 0.125 | 0.225 | 0.048  |
|                | Multiscale Mel   | VICReg   | 0.035 | 0.115 | 0.215 | 0.045  |
|                | Multiscale Mel   | Cosine   | 0.031 | 0.105 | 0.200 | 0.039  |
| Normal BEATS   | Normal Mel       | InfoNCE  | 0.120 | 0.328 | 0.470 | 0.202  |
|                | Normal Mel       | VICReg   | 0.110 | 0.312 | 0.450 | 0.186  |
|                | Normal Mel       | Cosine   | 0.107 | 0.297 | 0.426 | 0.174  |
