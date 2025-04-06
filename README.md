# Audio-Deepfake-Detection
Research & Model Selection

This section breaks down three promising ways to spot audio deepfakes, focusing on how technically strong they are, whether they can work in real time and how useful they are for catching AI-generated speech in actual conversations.

1. RawNet2
   
What makes it stand out:
It’s an end to end model that learns straight from raw audio waveforms. So unlike older methods, we don’t have to extract features like MFCCs manually ,it just learns what it needs on its own.

How it performs:
Achieved an Equal Error Rate (EER) of 1.85% on the ASVspoof2019 LA dataset.

Why it’s worth looking into:
No need for handcrafted features,
Handles a wide range of real-world audio inputs,
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

## Part 2: Implementation (Please Check code)

## Part 3: Documentation and Analysis

Implementation Process

Challenges I faced:
The main issue was figuring out how to get the audio into a format that a CNN can actually use. I started with raw audio, but then had to convert it into mel-spectrograms. That’s when I got a shape mismatch error, the audio files were all different lengths, and stacking them just didn’t work. Training also gave me a few problems, especially when the model didn’t like the input shapes or ran into memory issues.

How I fixed it:
I resized all the spectrograms to a fixed shape (128x400) so the model wouldn’t complain. Added padding where needed and normalized the inputs to keep things clean. For training I kept the model architecture simple and tweaked the batch size to make it easier on my system.

Stuff I assumed:
I went with the idea that mel-spectrograms are good enough to represent speech and catch deepfake patterns. Also assumed that the real vs fake labels in the dataset were accurate enough for this kind of experiment. And yeah that a basic CNN could learn some useful stuff without needing a super complex setup.

Analysis

Why I picked this model:
I went with a simple CNN using mel-spectrograms mainly because it was easier to get started with and didn’t need heavy compute. Since I’m still learning this stuff, I thought it made sense to start with something basic and build from there. It also kind of fits the idea of detecting deepfake speech in real-time. Plus I was curious to see how visual features from audio would work.

How it works (high-level):
Each audio file gets converted into a mel-spectrogram — basically a 2D image showing frequency over time. The CNN then tries to learn patterns in these spectrograms to figure out if the speech is real or fake. It's like teaching the model to look at audio.

Performance results:
After 10 epochs of training, I got a validation accuracy of around 84.62%, which I think is pretty solid for a simple setup without much tuning.

Strengths:
Quick to train,
Easy to understand what's going on,
Doesn’t need a huge dataset to start getting results


Weaknesses:
Might not work as well on real-life conversations,
If the fake audio is too good, the spectrogram might miss it,
Needs more training and fine-tuning to get better accuracy


Things to improve later:

I should try using a model that's already trained (so I don’t have to start from scratch),

Test out models that work directly with raw audio, like RawNet2,

Add more audio files or try tricks like changing pitch or adding background noise to make the model better,

Mix the spectrogram features with some info about the speaker or voice to make it smarter


## Setup Instructions
1. Clone the repo:  
   `git clone https://github.com/yourusername/your-repo-name`

2. Install dependencies:  
   `pip install -r requirements.txt`

3. Run the notebook or script in any Jupyter environment.
   
Note: You may need to download the dataset manually if not included.


