import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table
from IPython.display import display, Image
import io
import base64

# Create dataset table data with consistent lengths
data = {
    "Algorithm": [
        "RawNet2", "RawGAT-ST", "AASIST", "To-RawNet",
        "light-DARTS", "SpecRNet", "PS3DT", "HM-Conformer",
        "Multi-Fusion", "CD-ADD", "DSVAE", "ONE-CLASS",
        "RWM[16]", "RAD", "RawSpectrogram", "Real-Time"
    ],
    "Year": [
        "2020", "2021", "2022", "2023", "2022", "2021", "2024", "2023",
        "2024", "2024", "2023", "2024", "2024", "2024", "2023", "2024"
    ],
    "Features": [
        "Raw waveform", "Raw waveform", "Raw waveform", "Raw waveform",
        "wav2vec", "Spectrogram", "Spectrogram", "LFCC",
        "WavLM", "Wav2Vec2, Whisper", "Spectrogram", "wav2vec2",
        "wav2vec2", "WavLM", "Raw Spectrogram", "Raw waveform"
    ],
    "Classifier": [
        "RawNet2", "GAT-ST", "HS-GAL, MGO", "RawNet2, TCN",
        "light-DARTS", "SpecNet", "Transformer", "HM-Conformer",
        "MFA,ASP", "NN-based", "VAE", "AASIST",
        "S-CNN", "MFA", "ResNet, LCNN", "RECAPA TDNN"
    ],
    "Data Aug": [
        "×", "√ (channel mask)", "√ (RawBoost)", "×",
        "×", "×", "×", "×",
        "×", "√", "×", "√",
        "√", "√", "×", "×"
    ],
    "Extra Data": [
        "×", "√", "×", "×",
        "×", "√", "×", "√",
        "×", "×", "×", "√",
        "√", "√", "×", "×"
    ],
    "Notes": [
        "Requires self-supervised pre-training",
        "Extracts speaker features from raw waveform",
        "Generalizes to unknown synthesizers",
        "End-to-end detection of various spoofing attacks",
        "Reduces filter correlation via orthogonal conv",
        "Reduces manual parameter tuning impact",
        "Reduces memory consumption",
        "Improves detection via mel-spectrogram patches",
        "Hierarchical pooling and multi-level aggregation",
        "Cross-domain evaluation and few-shot learning",
        "One-class classification with teacher-student",
        "Adaptive gradient direction for generalization",
        "Retrieves similar samples to enhance detection",
        "Converts audio to spectrogram with RNN streaming",
        "Focuses on developing real-time detection",
        "Optimized for low-latency processing"
    ]
}

# Verify all columns have same length
assert all(len(v) == 16 for v in data.values()), "All columns must have same length"

# Create figure with larger size
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')
ax.axis('tight')

# Create table with improved formatting
table = ax.table(cellText=[list(data.keys())] + list(zip(*data.values())),
                loc='center',
                cellLoc='left',
                colLoc='center',
                colWidths=[0.12, 0.05, 0.1, 0.12, 0.08, 0.08, 0.35])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Apply cell styling
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor('gray')
    cell.set_linewidth(0.8)
    if i == 0:  # Header row
        cell.set_text_props(weight='bold', horizontalalignment='center')
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
        cell.set_facecolor('#f5f5f5')

# Add title
plt.title("Comparison of End-to-End Audio Deepfake Detection Methods", 
          y=1.02, fontsize=12, weight='bold')

# Save to buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0.5)
plt.close(fig)

# Display in notebook
img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
display(Image(url=f"data:image/png;base64,{img_data}"))

# Save to file
with open('audio_detection_comparison.png', 'wb') as f:
    f.write(buf.getvalue())
print("Table saved as 'audio_detection_comparison.png'")