import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import sys

MAX_FRAMES = 300

class CNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_and_trim(path, sr=22050):
    y, sr = librosa.load(path, sr=sr)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    return y_trimmed, sr

def wav_to_logmel(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    if logmel.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad_width)), mode="constant")
    else:
        logmel = logmel[:, :MAX_FRAMES]
    return logmel

def process_sample(path):
    y, sr = load_and_trim(path)
    return wav_to_logmel(y, sr)

def predict_emotion(wav_path, model_path='best_model.pth'):
    idx_to_emotion = {
        0: "neutral",
        1: "calm",
        2: "happy",
        3: "sad",
        4: "angry",
        5: "fearful",
        6: "disgust",
        7: "surprised"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN()
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    mel = process_sample(wav_path)
    mel_tensor = torch.tensor(mel, dtype=torch.float32)
    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)
    mel_tensor = mel_tensor.to(device)

    with torch.no_grad():
        output = model(mel_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        predicted_emotion = idx_to_emotion[predicted.item()]
        confidence_percentage = confidence.item() * 100

    print(f"Predicted Emotion: {predicted_emotion}")
    print(f"Confidence: {confidence_percentage:.2f}%")

    return predicted_emotion, confidence_percentage

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_wav_file> [model_path]")
        print("Example: python predict.py audio.wav")
        print("Example: python predict.py audio.wav best_model.pth")
        sys.exit(1)

    wav_file = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else 'best_model.pth'

    emotion, confidence = predict_emotion(wav_file, model_file)
