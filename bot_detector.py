#!/usr/bin/env python3
"""
Bot or Not Challenge - Maximum Accuracy Bot Detector
=====================================================
Full ML pipeline: feature extraction → model training → threshold optimization
Supports both English and French datasets.
"""

import json
import re
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(data_json):
    """
    Extract 30+ features per user from a dataset.
    Returns a DataFrame with user_id as index and all features as columns.
    """
    users = {u['id']: u for u in data_json['users']}
    posts = data_json['posts']
    lang = data_json['lang']

    # Group posts by user
    posts_by_user = defaultdict(list)
    for p in posts:
        ts = datetime.fromisoformat(p['created_at'].replace('Z', '+00:00'))
        posts_by_user[p['author_id']].append({
            'text': p['text'],
            'timestamp': ts,
            'lang': p.get('lang', ''),
        })

    # Sort posts by time for each user
    for uid in posts_by_user:
        posts_by_user[uid].sort(key=lambda x: x['timestamp'])

    features_list = []

    for uid, user in users.items():
        user_posts = posts_by_user.get(uid, [])
        texts = [p['text'] for p in user_posts]
        timestamps = [p['timestamp'] for p in user_posts]

        f = {'user_id': uid}

        # =====================================================================
        # 1. BASIC USER PROFILE FEATURES
        # =====================================================================
        f['tweet_count'] = user.get('tweet_count', 0)
        f['z_score'] = user.get('z_score', 0.0)

        # Username features
        username = user.get('username', '')
        f['username_length'] = len(username)
        f['username_num_digits'] = len(re.findall(r'\d', username))
        f['username_has_numbers'] = 1 if re.search(r'\d', username) else 0
        f['username_num_underscores'] = username.count('_')
        f['username_has_underscore'] = 1 if '_' in username else 0
        f['username_is_capitalized'] = 1 if username[0].isupper() else 0 if username else 0
        f['username_all_lower'] = 1 if username == username.lower() else 0
        f['username_digit_ratio'] = len(re.findall(r'\d', username)) / max(len(username), 1)

        # Name features
        name = user.get('name', '') or ''
        f['name_length'] = len(name)
        f['name_has_emoji'] = 1 if any(ord(c) > 8000 for c in name) else 0
        f['name_num_words'] = len(name.split())
        f['name_has_numbers'] = 1 if re.search(r'\d', name) else 0

        # Description features
        desc = user.get('description', '') or ''
        f['desc_length'] = len(desc)
        f['desc_is_empty'] = 1 if not desc.strip() else 0
        f['desc_num_words'] = len(desc.split()) if desc.strip() else 0
        f['desc_has_emoji'] = 1 if any(ord(c) > 8000 for c in desc) else 0
        f['desc_has_hashtag'] = 1 if '#' in desc else 0
        f['desc_has_url'] = 1 if 'http' in desc or 't.co' in desc else 0
        f['desc_has_mention'] = 1 if '@' in desc else 0

        # Location features
        loc = user.get('location', '') or ''
        f['location_is_empty'] = 1 if not loc.strip() else 0
        f['location_length'] = len(loc)

        # =====================================================================
        # 2. POSTING TIMING FEATURES
        # =====================================================================
        if len(timestamps) >= 2:
            gaps = [(timestamps[i+1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps)-1)]
            gaps = [g for g in gaps if g >= 0]  # safety

            if gaps:
                f['timing_mean_gap'] = np.mean(gaps)
                f['timing_std_gap'] = np.std(gaps)
                f['timing_min_gap'] = np.min(gaps)
                f['timing_max_gap'] = np.max(gaps)
                f['timing_median_gap'] = np.median(gaps)
                f['timing_cv'] = np.std(gaps) / max(np.mean(gaps), 1)  # regularity

                # Quartile-based features
                f['timing_q25'] = np.percentile(gaps, 25)
                f['timing_q75'] = np.percentile(gaps, 75)
                f['timing_iqr'] = f['timing_q75'] - f['timing_q25']

                # Burstiness: how many gaps < 60 seconds
                f['timing_burst_count'] = sum(1 for g in gaps if g < 60)
                f['timing_burst_ratio'] = f['timing_burst_count'] / len(gaps)

                # Very regular posting detection
                f['timing_very_regular'] = 1 if f['timing_cv'] < 0.5 else 0

                # Gaps that are suspiciously uniform
                if len(gaps) >= 3:
                    sorted_gaps = sorted(gaps)
                    # Check if gaps cluster around specific intervals
                    gap_diffs = [abs(sorted_gaps[i+1] - sorted_gaps[i]) for i in range(len(sorted_gaps)-1)]
                    f['timing_gap_uniformity'] = np.std(gaps) / max(np.mean(gaps), 1)
                else:
                    f['timing_gap_uniformity'] = 0
            else:
                for k in ['timing_mean_gap', 'timing_std_gap', 'timing_min_gap',
                          'timing_max_gap', 'timing_median_gap', 'timing_cv',
                          'timing_q25', 'timing_q75', 'timing_iqr',
                          'timing_burst_count', 'timing_burst_ratio',
                          'timing_very_regular', 'timing_gap_uniformity']:
                    f[k] = 0
        else:
            for k in ['timing_mean_gap', 'timing_std_gap', 'timing_min_gap',
                      'timing_max_gap', 'timing_median_gap', 'timing_cv',
                      'timing_q25', 'timing_q75', 'timing_iqr',
                      'timing_burst_count', 'timing_burst_ratio',
                      'timing_very_regular', 'timing_gap_uniformity']:
                f[k] = 0

        # Hour of day distribution
        if timestamps:
            hours = [t.hour for t in timestamps]
            hour_counts = Counter(hours)
            f['hour_unique_count'] = len(hour_counts)
            f['hour_entropy'] = -sum((c/len(hours)) * np.log2(c/len(hours))
                                     for c in hour_counts.values() if c > 0)
            f['hour_mode_ratio'] = max(hour_counts.values()) / len(hours)

            # Night posting (0-6 AM)
            night_posts = sum(1 for h in hours if 0 <= h <= 6)
            f['night_post_ratio'] = night_posts / len(hours)

            # Active hours span
            f['hour_span'] = max(hours) - min(hours) if hours else 0

            # Day distribution
            days = [t.date() for t in timestamps]
            day_counts = Counter(days)
            f['day_count'] = len(day_counts)
            if len(day_counts) > 1:
                day_vals = list(day_counts.values())
                f['day_balance'] = min(day_vals) / max(day_vals)
            else:
                f['day_balance'] = 1.0
        else:
            f['hour_unique_count'] = 0
            f['hour_entropy'] = 0
            f['hour_mode_ratio'] = 0
            f['night_post_ratio'] = 0
            f['hour_span'] = 0
            f['day_count'] = 0
            f['day_balance'] = 0

        # =====================================================================
        # 3. TEXT CONTENT FEATURES
        # =====================================================================
        if texts:
            # Basic text stats
            text_lengths = [len(t) for t in texts]
            f['text_avg_length'] = np.mean(text_lengths)
            f['text_std_length'] = np.std(text_lengths)
            f['text_min_length'] = np.min(text_lengths)
            f['text_max_length'] = np.max(text_lengths)

            # Word count per tweet
            word_counts = [len(t.split()) for t in texts]
            f['text_avg_words'] = np.mean(word_counts)
            f['text_std_words'] = np.std(word_counts)

            # Vocabulary diversity
            all_words = ' '.join(texts).lower().split()
            f['vocab_size'] = len(set(all_words))
            f['vocab_total'] = len(all_words)
            f['vocab_diversity'] = len(set(all_words)) / max(len(all_words), 1)

            # Hapax legomena ratio (words used only once)
            word_freq = Counter(all_words)
            f['hapax_ratio'] = sum(1 for w, c in word_freq.items() if c == 1) / max(len(word_freq), 1)

            # Type-token ratio in sliding windows (more robust)
            if len(all_words) >= 50:
                ttr_windows = []
                window_size = 50
                for i in range(0, len(all_words) - window_size + 1, 25):
                    window = all_words[i:i+window_size]
                    ttr_windows.append(len(set(window)) / window_size)
                f['vocab_ttr_windowed'] = np.mean(ttr_windows)
            else:
                f['vocab_ttr_windowed'] = f['vocab_diversity']

            # Hashtag features
            hashtag_counts = [t.count('#') for t in texts]
            f['hashtag_per_tweet'] = np.mean(hashtag_counts)
            f['hashtag_total'] = sum(hashtag_counts)
            f['tweets_with_hashtag_ratio'] = sum(1 for c in hashtag_counts if c > 0) / len(texts)

            # Extract actual hashtags
            all_hashtags = re.findall(r'#\w+', ' '.join(texts).lower())
            f['unique_hashtags'] = len(set(all_hashtags))
            f['hashtag_diversity'] = len(set(all_hashtags)) / max(len(all_hashtags), 1) if all_hashtags else 0

            # URL features
            url_counts = [1 if ('http' in t or 't.co' in t) else 0 for t in texts]
            f['url_ratio'] = np.mean(url_counts)

            # Mention features
            mention_counts = [t.count('@') for t in texts]
            f['mention_per_tweet'] = np.mean(mention_counts)
            f['tweets_with_mention_ratio'] = sum(1 for c in mention_counts if c > 0) / len(texts)

            # Duplicate/similarity detection
            unique_texts = len(set(texts))
            f['duplicate_ratio'] = 1 - unique_texts / len(texts)

            # Near-duplicate detection (first 30 chars)
            prefixes = [t[:30].lower() for t in texts]
            unique_prefixes = len(set(prefixes))
            f['near_duplicate_ratio'] = 1 - unique_prefixes / len(texts)

            # Emoji usage
            emoji_count = sum(1 for t in texts for c in t if ord(c) > 8000)
            f['emoji_per_tweet'] = emoji_count / len(texts)
            f['tweets_with_emoji_ratio'] = sum(1 for t in texts if any(ord(c) > 8000 for c in t)) / len(texts)

            # Punctuation features
            f['exclamation_per_tweet'] = sum(t.count('!') for t in texts) / len(texts)
            f['question_per_tweet'] = sum(t.count('?') for t in texts) / len(texts)
            f['ellipsis_per_tweet'] = sum(t.count('...') for t in texts) / len(texts)

            # Capitalization patterns
            f['all_caps_words_ratio'] = sum(1 for w in all_words if w.isupper() and len(w) > 1) / max(len(all_words), 1)

            # Average sentence length (rough proxy via periods)
            period_count = sum(t.count('.') for t in texts)
            f['avg_sentences_per_tweet'] = period_count / len(texts)

            # Newline usage
            f['newline_per_tweet'] = sum(t.count('\n') for t in texts) / len(texts)

            # Text similarity between consecutive tweets
            if len(texts) >= 2:
                def jaccard(a, b):
                    sa, sb = set(a.lower().split()), set(b.lower().split())
                    if not sa or not sb:
                        return 0
                    return len(sa & sb) / len(sa | sb)

                consecutive_sims = [jaccard(texts[i], texts[i+1]) for i in range(len(texts)-1)]
                f['consecutive_similarity_mean'] = np.mean(consecutive_sims)
                f['consecutive_similarity_max'] = np.max(consecutive_sims)
                f['consecutive_similarity_std'] = np.std(consecutive_sims)
            else:
                f['consecutive_similarity_mean'] = 0
                f['consecutive_similarity_max'] = 0
                f['consecutive_similarity_std'] = 0

            # Topic diversity (how many different topics does user tweet about)
            # Use keyword matching from metadata
            topics = data_json.get('metadata', {}).get('topics', [])
            topic_hits = defaultdict(int)
            for t in texts:
                t_lower = t.lower()
                for topic in topics:
                    for kw in topic.get('keywords', []):
                        if kw.lower() in t_lower:
                            topic_hits[topic['topic']] += 1
                            break
            f['topic_count'] = len(topic_hits)
            f['topic_tweet_ratio'] = sum(topic_hits.values()) / len(texts) if topic_hits else 0

            # Retweet-like patterns
            f['rt_ratio'] = sum(1 for t in texts if t.lower().startswith('rt ') or t.lower().startswith('rt:')) / len(texts)

        else:
            # No posts - fill with zeros
            for k in ['text_avg_length', 'text_std_length', 'text_min_length', 'text_max_length',
                      'text_avg_words', 'text_std_words', 'vocab_size', 'vocab_total',
                      'vocab_diversity', 'hapax_ratio', 'vocab_ttr_windowed',
                      'hashtag_per_tweet', 'hashtag_total', 'tweets_with_hashtag_ratio',
                      'unique_hashtags', 'hashtag_diversity',
                      'url_ratio', 'mention_per_tweet', 'tweets_with_mention_ratio',
                      'duplicate_ratio', 'near_duplicate_ratio',
                      'emoji_per_tweet', 'tweets_with_emoji_ratio',
                      'exclamation_per_tweet', 'question_per_tweet', 'ellipsis_per_tweet',
                      'all_caps_words_ratio', 'avg_sentences_per_tweet', 'newline_per_tweet',
                      'consecutive_similarity_mean', 'consecutive_similarity_max',
                      'consecutive_similarity_std', 'topic_count', 'topic_tweet_ratio',
                      'rt_ratio']:
                f[k] = 0

        features_list.append(f)

    df = pd.DataFrame(features_list)
    df = df.set_index('user_id')
    return df


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def competition_score(y_true, y_pred):
    """Calculate the competition score: +4 TP, -1 FN, -2 FP"""
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    score = 4 * tp - 1 * fn - 2 * fp
    max_possible = 4 * sum(y_true)
    return score, tp, fn, fp, tn, max_possible


def find_optimal_threshold(y_true, y_proba):
    """Find the threshold that maximizes competition score."""
    best_threshold = 0.5
    best_score = -999999

    for threshold in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        score, tp, fn, fp, tn, _ = competition_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def train_and_evaluate(train_features, train_labels, test_features=None, test_labels=None, lang="en"):
    """
    Train an ensemble model and evaluate performance.
    """
    feature_cols = [c for c in train_features.columns if c != 'is_bot']

    X_train = train_features[feature_cols].values
    y_train = train_labels.values

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Build ensemble of 3 strong models
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )

    # Soft voting ensemble
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[2, 1, 1],  # Give XGBoost more weight
    )

    # --- Cross-validation on training data ---
    print(f"\n{'='*60}")
    print(f"TRAINING ({lang.upper()}) - {len(X_train)} users, {sum(y_train)} bots")
    print(f"{'='*60}")

    # Get cross-validated probabilities
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probas = cross_val_predict(ensemble, X_train_scaled, y_train, cv=cv, method='predict_proba')[:, 1]

    # Find optimal threshold on CV predictions
    best_thresh, best_cv_score = find_optimal_threshold(y_train, cv_probas)
    cv_preds = (cv_probas >= best_thresh).astype(int)
    score, tp, fn, fp, tn, max_possible = competition_score(y_train, cv_preds)

    print(f"\nCross-Validation Results (optimal threshold={best_thresh:.2f}):")
    print(f"  Competition Score: {score} / {max_possible} possible")
    print(f"  TP={tp} (bots caught), FN={fn} (bots missed), FP={fp} (humans wrongly flagged), TN={tn}")
    print(f"  Precision: {tp/max(tp+fp,1):.3f}, Recall: {tp/max(tp+fn,1):.3f}")

    # Train final model on all training data
    ensemble.fit(X_train_scaled, y_train)

    # Feature importance (from XGBoost)
    xgb_model = ensemble.named_estimators_['xgb']
    importances = xgb_model.feature_importances_
    top_features = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:15]
    print(f"\nTop 15 Features:")
    for fname, imp in top_features:
        print(f"  {fname}: {imp:.4f}")

    # --- Test on held-out dataset if provided ---
    if test_features is not None and test_labels is not None:
        X_test = test_features[feature_cols].values
        y_test = test_labels.values
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
        X_test_scaled = scaler.transform(X_test)

        test_probas = ensemble.predict_proba(X_test_scaled)[:, 1]

        # Use the threshold optimized on training CV
        test_preds = (test_probas >= best_thresh).astype(int)
        score, tp, fn, fp, tn, max_possible = competition_score(y_test, test_preds)

        print(f"\nHeld-Out Test Results (threshold={best_thresh:.2f}):")
        print(f"  Competition Score: {score} / {max_possible} possible")
        print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        print(f"  Precision: {tp/max(tp+fp,1):.3f}, Recall: {tp/max(tp+fn,1):.3f}")

        # Also try optimizing threshold on test data to see ceiling
        best_test_thresh, best_test_score = find_optimal_threshold(y_test, test_probas)
        test_preds_opt = (test_probas >= best_test_thresh).astype(int)
        score_opt, tp2, fn2, fp2, tn2, _ = competition_score(y_test, test_preds_opt)
        print(f"\n  (Oracle test threshold={best_test_thresh:.2f} would give score={score_opt})")

    return ensemble, scaler, best_thresh, feature_cols


def predict_bots(model, scaler, threshold, features_df, feature_cols):
    """Predict bot IDs from a new dataset."""
    X = features_df[feature_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_scaled = scaler.transform(X)
    probas = model.predict_proba(X_scaled)[:, 1]
    bot_mask = probas >= threshold
    bot_ids = features_df.index[bot_mask].tolist()
    return bot_ids, probas


# =============================================================================
# MAIN
# =============================================================================

def load_dataset(json_path):
    """Load a dataset JSON file."""
    with open(json_path) as f:
        return json.load(f)

def load_bot_labels(txt_path):
    """Load bot labels from a txt file."""
    with open(txt_path) as f:
        return set(line.strip() for line in f if line.strip())


if __name__ == '__main__':

    if len(sys.argv) >= 2 and sys.argv[1] == 'detect':
        # === DETECTION MODE: Run on new dataset ===
        if len(sys.argv) < 3:
            print("Usage: python3 bot_detector.py detect <dataset.json> [model_dir]")
            sys.exit(1)

        dataset_path = sys.argv[2]
        model_dir = sys.argv[3] if len(sys.argv) > 3 else 'models'

        data = load_dataset(dataset_path)
        lang = data['lang']

        print(f"Detecting bots in {dataset_path} ({lang})...")
        features = extract_features(data)

        # Load model
        import pickle
        model_path = os.path.join(model_dir, f'model_{lang}.pkl')
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)

        model = saved['model']
        scaler = saved['scaler']
        threshold = saved['threshold']
        feature_cols = saved['feature_cols']

        bot_ids, probas = predict_bots(model, scaler, threshold, features, feature_cols)

        # Output
        output_file = f"detections.{lang}.txt"
        with open(output_file, 'w') as f:
            for bid in bot_ids:
                f.write(bid + '\n')

        print(f"Found {len(bot_ids)} bots out of {len(features)} users")
        print(f"Results written to {output_file}")

    else:
        # === TRAINING MODE: Train and evaluate on practice datasets ===
        base_dir = 'datasets'

        # Load all datasets
        print("Loading datasets...")
        datasets = {}
        for num in [30, 31, 32, 33]:
            json_path = os.path.join(base_dir, f'dataset.posts&users.{num}.json')
            bots_path = os.path.join(base_dir, f'dataset.bots.{num}.txt')
            data = load_dataset(json_path)
            bots = load_bot_labels(bots_path)

            print(f"  Extracting features for dataset {num} ({data['lang']})...")
            features = extract_features(data)
            labels = pd.Series([1 if uid in bots else 0 for uid in features.index],
                             index=features.index, name='is_bot')

            datasets[num] = {
                'data': data,
                'bots': bots,
                'features': features,
                'labels': labels,
                'lang': data['lang'],
            }
            print(f"    {len(features)} users, {sum(labels)} bots, {len(features.columns)} features")

        # =====================================================================
        # ENGLISH: Train on 30, test on 32 (and vice versa), then train on both
        # =====================================================================
        print("\n" + "="*60)
        print("ENGLISH DETECTOR")
        print("="*60)

        # Train on 30, test on 32
        print("\n--- Train on Dataset 30, Test on Dataset 32 ---")
        model_30, scaler_30, thresh_30, fcols_30 = train_and_evaluate(
            datasets[30]['features'], datasets[30]['labels'],
            datasets[32]['features'], datasets[32]['labels'],
            lang='en'
        )

        # Train on 32, test on 30
        print("\n--- Train on Dataset 32, Test on Dataset 30 ---")
        model_32, scaler_32, thresh_32, fcols_32 = train_and_evaluate(
            datasets[32]['features'], datasets[32]['labels'],
            datasets[30]['features'], datasets[30]['labels'],
            lang='en'
        )

        # Train on BOTH for final model
        print("\n--- Final English Model: Train on 30+32 ---")
        en_features = pd.concat([datasets[30]['features'], datasets[32]['features']])
        en_labels = pd.concat([datasets[30]['labels'], datasets[32]['labels']])
        model_en, scaler_en, thresh_en, fcols_en = train_and_evaluate(
            en_features, en_labels, lang='en'
        )

        # =====================================================================
        # FRENCH: Train on 31, test on 33 (and vice versa), then train on both
        # =====================================================================
        print("\n" + "="*60)
        print("FRENCH DETECTOR")
        print("="*60)

        # Train on 31, test on 33
        print("\n--- Train on Dataset 31, Test on Dataset 33 ---")
        model_31, scaler_31, thresh_31, fcols_31 = train_and_evaluate(
            datasets[31]['features'], datasets[31]['labels'],
            datasets[33]['features'], datasets[33]['labels'],
            lang='fr'
        )

        # Train on 33, test on 31
        print("\n--- Train on Dataset 33, Test on Dataset 31 ---")
        model_33, scaler_33, thresh_33, fcols_33 = train_and_evaluate(
            datasets[33]['features'], datasets[33]['labels'],
            datasets[31]['features'], datasets[31]['labels'],
            lang='fr'
        )

        # Train on BOTH for final model
        print("\n--- Final French Model: Train on 31+33 ---")
        fr_features = pd.concat([datasets[31]['features'], datasets[33]['features']])
        fr_labels = pd.concat([datasets[31]['labels'], datasets[33]['labels']])
        model_fr, scaler_fr, thresh_fr, fcols_fr = train_and_evaluate(
            fr_features, fr_labels, lang='fr'
        )

        # =====================================================================
        # SAVE MODELS
        # =====================================================================
        import pickle
        os.makedirs('models', exist_ok=True)

        for lang, model, scaler, thresh, fcols in [
            ('en', model_en, scaler_en, thresh_en, fcols_en),
            ('fr', model_fr, scaler_fr, thresh_fr, fcols_fr),
        ]:
            model_path = os.path.join('models', f'model_{lang}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'threshold': thresh,
                    'feature_cols': fcols,
                }, f)
            print(f"\nSaved {lang} model to {model_path} (threshold={thresh:.2f})")

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print("\nTo detect bots on a new dataset:")
        print("  python3 bot_detector.py detect <dataset.json>")
