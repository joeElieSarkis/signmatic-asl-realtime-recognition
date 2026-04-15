# SignMatic: Real-Time ASL Recognition System (MediaPipe + LSTM)

A real-time American Sign Language (ASL) recognition system that uses MediaPipe keypoints and an LSTM model to classify isolated signs and convert them into on-screen text (and later speech) for assistive communication.

## Current Approach
- MediaPipe Holistic for keypoint extraction (pose + hands)
- Sequence-based LSTM model for temporal gesture recognition
- Hybrid dataset:
  - WLASL videos for base training
  - Custom-recorded data for fine-tuning
- Real-time inference with:
  - Idle class (no sign detection)
  - Hand presence gating
  - Confidence thresholding
  - Prediction cooldown

## Goal
Deploy a stable, real-time ASL recognition system on an embedded platform (Jetson) that outputs text and speech for a constrained vocabulary (15–30 words).

## Acknowledgment

This project was initially inspired by a public tutorial on action detection using MediaPipe and LSTM.
The implementation has been significantly extended and modified to support a full real-time ASL recognition pipeline, including custom dataset creation, WLASL-based training, and embedded deployment.