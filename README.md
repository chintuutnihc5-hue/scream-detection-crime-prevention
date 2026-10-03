# Human Scream Detection for Crime Prevention

This project showcases a practical system for detecting human screams from audio and using the results to support crime prevention efforts. The repository includes:

- an audio feature pipeline for scream detection
- a lightweight ML-inspired heuristic detector that works without live microphone hardware
- a Plotly Dash dashboard for monitoring alerts, incident trends, and risk levels
- a structure ready to evolve into a CNN/Deep Learning pipeline with real surveillance audio data

## Project goals

- Detect scream-like audio patterns in real time
- Reduce false alarms by combining audio features and context
- Visualize incident activity for monitoring and prevention
- Create a usable project foundation for academic, research, or prototype deployment

## Included components

- `src/scream_detection.py` – audio generation, feature extraction, and scream scoring logic
- `app.py` – an interactive dashboard for monitoring incidents and alerts
- `requirements.txt` – project dependencies
- `README.md` – usage and project overview

## Quick start

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the dashboard:

   ```bash
   python app.py
   ```

5. Open the local Dash URL in your browser.

## Example detection run

```bash
python
from src.scream_detection import generate_scream_wav, detect_scream

generate_scream_wav("data/scream_sample.wav")
print(detect_scream("data/scream_sample.wav"))
```

## Notes

- This repository uses a synthetic scream-like waveform for testing and demo purposes.
- For real-world deployment, replace the heuristic model with a labelled dataset and a trained CNN or transfer learning model such as YAMNet.
- Privacy and compliance should be reviewed before deploying this in public or surveillance environments.

## License

This project is intended for educational and prototype use.
