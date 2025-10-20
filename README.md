# Audio-Text Retrieval System

A deep learning system that retrieves audio files based on natural language queries. Simply describe the sound you're looking for in text, and the system will find and rank the most relevant audio clips from a dataset.

## Overview

This project implements a bi-modal retrieval system that bridges the gap between audio and text. Given a text query like "dog barking in the distance" or "rain falling on a metal roof," the system encodes both the text and audio files into a shared embedding space, then retrieves and ranks audio files based on their semantic similarity to the query.

The system supports multiple state-of-the-art audio encoder architectures and loss functions, making it a flexible framework for audio-text retrieval research and applications.



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
- R@1: Recall score for the top-1 retrieved result
- R@5: Recall among the top-5 retrieved results
- R@10: Recall among the top-10 retrieved results
- mAP@10: Mean Average Precision across the top-10 results


