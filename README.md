# AI Digit Recognizer ✍️🤖

A small, complete machine-learning project: draw a digit (0–9) in your browser and an AI model guesses which digit it is, with a confidence score for every option.

![Screenshot of the app](docs/screenshot.png)

## What it does

- **Trains** a Support Vector Machine (SVM) on 1,797 real handwritten digits (the dataset ships with scikit-learn, so no download is needed).
- **Scores 98% accuracy** on a held-out test set the model never saw during training.
- **Serves a web app** (Flask) with a drawing canvas. Your drawing is converted into the same 8×8 format as the training data, then classified.
- **Includes tests** for the model, the image preprocessing and the web API.

## Project structure

```
digit-recognizer-ai/
├── app.py                  # Flask web server (the app you open in the browser)
├── templates/index.html    # Drawing canvas + results page
├── src/
│   ├── train.py            # Trains the model and saves it to models/
│   ├── predict.py          # Predicts a digit from an image file (command line)
│   └── preprocess.py       # Turns a drawing into the model's 8x8 input format
├── models/digit_model.joblib  # The trained model (ready to use)
├── tests/test_project.py   # Automated tests (pytest)
├── docs/screenshot.png
├── requirements.txt
└── README.md
```

## Quick start

You need **Python 3.9 or newer**.

```bash
# 1. Get the code
git clone https://github.com/<your-username>/digit-recognizer-ai.git
cd digit-recognizer-ai

# 2. Create a virtual environment and install dependencies
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the web app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser, draw a digit and press **Predict**.

## Other commands

| What | Command |
|---|---|
| Retrain the model (prints accuracy report) | `python src/train.py` |
| Predict a digit from an image file | `python src/predict.py my_digit.png` |
| Run the tests | `python -m pytest` |

## How it works

1. **Data** – The scikit-learn *digits* dataset: 8×8 grayscale images, each pixel a value from 0 to 16.
2. **Model** – Pixels are standardised (`StandardScaler`), then classified by an SVM with an RBF kernel. The SVM is wrapped in `CalibratedClassifierCV` so it can report a probability for each digit.
3. **Preprocessing your drawing** – The canvas image is cropped to the ink, centred in a square with a small border, reduced to a 32×32 black-and-white image, and then counted in 4×4 blocks to give an 8×8 grid with values 0–16. This is the same recipe used to build the original dataset, which is why drawings from the browser are recognised reliably.

## Tips for best results

- Draw one digit, large, filling most of the box.
- Press **Clear** before drawing the next digit.

## Ideas to extend it

- Train on the larger MNIST dataset (28×28 images) with a convolutional neural network.
- Deploy it online (Render, Railway or Hugging Face Spaces).
- Let users correct wrong predictions and collect those as new training data.

## License

MIT – see [LICENSE](LICENSE).
