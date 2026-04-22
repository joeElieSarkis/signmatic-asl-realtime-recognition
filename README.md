# SignMatic: Real-Time ASL Recognition System (MediaPipe + LSTM)

A real-time American Sign Language (ASL) recognition system that uses MediaPipe keypoints and an LSTM model to classify isolated signs and convert them into on-screen text, with speech output support for assistive communication.

## Current Approach

- MediaPipe Holistic for keypoint extraction from pose and hands
- Sequence-based LSTM model for temporal gesture recognition
- Hybrid dataset:
  - MS-ASL clips for base training
  - Custom-recorded data for vocabulary expansion and adaptation
- Real-time inference with:
  - Idle class for no-sign state
  - Hand presence gating
  - Confidence thresholding
  - Prediction stabilization
  - Cooldown between accepted predictions

## Project Structure

- `src/custom/`  
  Custom data collection, dataset building, model training, and realtime inference

- `src/msasl/`  
  MS-ASL metadata export, video download, clip extraction, keypoint extraction, and dataset preparation

- `src/hybrid/`  
  Hybrid dataset building, model training, and realtime inference using combined MS-ASL and custom data

- `data/`  
  Local datasets, extracted keypoints, and processed training arrays  
  Not included in the repository

- `models/`  
  Saved trained model files  
  Generated locally and not included in the repository

- `outputs/`  
  Training logs and experiment outputs  
  Generated locally and not included in the repository

- `notebooks/`  
  Early experimentation notebooks based on the initial tutorial workflow

## Goal

Deploy a stable, real-time ASL recognition system on an embedded platform such as Jetson, where a user can stand in front of the system, perform supported signs, and receive text and speech output in real time.

## Current Scope

The system currently focuses on isolated-word recognition with a constrained vocabulary. The vocabulary is being expanded incrementally through hybrid training with MS-ASL and custom data to improve real-time usability and generalization.

## Acknowledgment

This project was initially inspired by a public tutorial on action detection using MediaPipe and LSTM. The implementation has since been substantially redesigned and extended into a full ASL recognition pipeline with custom data collection, MS-ASL integration, hybrid training, realtime inference logic, and planned embedded deployment.