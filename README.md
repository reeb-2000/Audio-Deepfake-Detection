# Audio-Deepfake-Detection
Research & Model Selection

This section breaks down three promising ways to spot audio deepfakes, focusing on how technically strong they are, whether they can work in real time and how useful they are for catching AI-generated speech in actual conversations.

1. RawNet2
   
What makes it stand out:
It’s an end to end model that learns straight from raw audio waveforms. So unlike older methods, we don’t have to extract features like MFCCs manually ,it just learns what it needs on its own.

How it performs:
Achieved an Equal Error Rate (EER) of 1.85% on the ASVspoof2019 LA dataset.

Why it’s worth looking into:
No need for handcrafted features
Handles a wide range of real-world audio inputs
Could be adapted for real-time use

Limitations:
Needs a lot of data to train effectively,
slightly heavy model, so it might need a bit of tuning to work smoothly.

2. LCNN (Lightweight CNN)
   
What makes it stand out:
It uses something called max-feature-map activations, which help highlight the most important parts of the audio signal. It works well with features like MFCCs or CQCCs.

How it performs:
Reported an EER of 2.7% on ASVspoof benchmark datasets.

Why it’s worth looking into:
Fast and lightweight ,good option for near real-time scenarios,
Performs well even when audio is short or noisy

Limitations:
Relies a lot on preprocessed features,
Might need a bit of fine-tuning to spot the trickier deepfakes.

3. Spectrogram-Based CNN (My Implementation)
   
What makes it stand out:
We convert the audio into Log-Mel spectrograms and feed those into a CNN to classify real vs. fake speech. It’s simple and gets the job done.

How it performs:
In our own experiments, we got a validation accuracy of 84.62%.

Why it’s worth looking into:
Straightforward, easy to understand and implement,
Quick to train and deploy,
Works well for distinguishing real from synthetic audio clips

Limitations:
The results depend a lot on how we set the spectrogram parameters,
May not generalize well across every kind of AI-generated voice
