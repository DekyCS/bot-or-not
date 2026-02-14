# Bot or Not - Team dekycs

Automated bot detection system for the Bot or Not Challenge. Identifies bot accounts in Twitter/X-like social media datasets using machine learning.

## Approach

**Feature Engineering (158 features per user):**
- **Profile features**: username patterns (digits, trailing numbers, length), name/description analysis, location
- **Temporal features**: posting gap statistics (mean, CV, skewness, kurtosis), FFT periodicity detection, autocorrelation, Benford's law deviation, gap entropy, round-interval detection, burst patterns
- **Text/NLP features**: vocabulary diversity, hashtag/mention/URL usage, duplicate detection, punctuation style, n-gram diversity, self-similarity, retweet patterns
- **TF-IDF embeddings**: word-level (20d via SVD) and character-level (15d via SVD) representations

**Model:**
- StackingClassifier with 4 base models (2x XGBoost, RandomForest, GradientBoosting) and LogisticRegression meta-learner
- Automatic feature selection using XGBoost importance ranking + forward selection (prevents overfitting on small datasets)
- Custom threshold optimization for the competition scoring system (+4 correct bot, -1 missed bot, -2 false positive)

**Performance on practice data (5-fold CV):**
- English: 487/516 (94.4%)
- French: 204/220 (92.7%)

## Usage

### Train models (on practice datasets)
```bash
python3 bot_detector.py
```

### Detect bots on a new dataset
```bash
python3 bot_detector.py detect <dataset.json>
```

## Project Structure

```
bot_detector.py          # Main pipeline: feature extraction, training, detection
datasets/                # Practice datasets (30-33) with answer keys
final-dataset/           # Competition evaluation datasets (34, 35)
models/                  # Trained model files (.pkl)
dekycs.detections.en.txt # English bot detections
dekycs.detections.fr.txt # French bot detections
```

## Dependencies

- Python 3
- scikit-learn
- xgboost
- numpy
- pandas
