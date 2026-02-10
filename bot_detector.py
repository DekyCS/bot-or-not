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
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(data_json):
    """
    Extract 120+ features per user from a dataset.
    Includes: profile, timing, text, NLP (TF-IDF), advanced temporal patterns.
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

    # =========================================================================
    # PRE-COMPUTE: TF-IDF on all users' combined texts (NLP features)
    # =========================================================================
    user_ids_ordered = list(users.keys())
    user_combined_texts = []
    for uid in user_ids_ordered:
        user_posts_list = posts_by_user.get(uid, [])
        combined = ' '.join(p['text'] for p in user_posts_list)
        user_combined_texts.append(combined if combined else '')

    # Word-level TF-IDF
    tfidf_word = TfidfVectorizer(
        max_features=500, ngram_range=(1, 2), min_df=2, max_df=0.95,
        sublinear_tf=True, strip_accents='unicode'
    )
    try:
        tfidf_word_matrix = tfidf_word.fit_transform(user_combined_texts)
        # Reduce to 20 dimensions with SVD
        svd_word = TruncatedSVD(n_components=20, random_state=42)
        tfidf_word_reduced = svd_word.fit_transform(tfidf_word_matrix)
    except:
        tfidf_word_reduced = np.zeros((len(user_ids_ordered), 20))

    # Character-level TF-IDF (catches stylistic patterns)
    tfidf_char = TfidfVectorizer(
        max_features=300, analyzer='char_wb', ngram_range=(3, 5),
        min_df=2, max_df=0.95, sublinear_tf=True
    )
    try:
        tfidf_char_matrix = tfidf_char.fit_transform(user_combined_texts)
        svd_char = TruncatedSVD(n_components=15, random_state=42)
        tfidf_char_reduced = svd_char.fit_transform(tfidf_char_matrix)
    except:
        tfidf_char_reduced = np.zeros((len(user_ids_ordered), 15))

    # Map uid -> index for TF-IDF lookup
    uid_to_idx = {uid: i for i, uid in enumerate(user_ids_ordered)}

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
        # Trailing digits (like "user123")
        trailing = re.search(r'\d+$', username)
        f['username_trailing_digits'] = len(trailing.group()) if trailing else 0

        # Name features
        name = user.get('name', '') or ''
        f['name_length'] = len(name)
        f['name_has_emoji'] = 1 if any(ord(c) > 8000 for c in name) else 0
        f['name_num_words'] = len(name.split())
        f['name_has_numbers'] = 1 if re.search(r'\d', name) else 0
        # Name-username similarity
        f['name_username_match'] = 1 if name.lower().replace(' ', '') == username.lower().replace('_', '') else 0

        # Description features
        desc = user.get('description', '') or ''
        f['desc_length'] = len(desc)
        f['desc_is_empty'] = 1 if not desc.strip() else 0
        f['desc_num_words'] = len(desc.split()) if desc.strip() else 0
        f['desc_has_emoji'] = 1 if any(ord(c) > 8000 for c in desc) else 0
        f['desc_has_hashtag'] = 1 if '#' in desc else 0
        f['desc_has_url'] = 1 if 'http' in desc or 't.co' in desc else 0
        f['desc_has_mention'] = 1 if '@' in desc else 0
        # Description vocab richness
        desc_words = desc.lower().split()
        f['desc_vocab_diversity'] = len(set(desc_words)) / max(len(desc_words), 1)

        # Location features
        loc = user.get('location', '') or ''
        f['location_is_empty'] = 1 if not loc.strip() else 0
        f['location_length'] = len(loc)

        # =====================================================================
        # 2. POSTING TIMING FEATURES (ENHANCED)
        # =====================================================================
        TIMING_ZERO_KEYS = [
            'timing_mean_gap', 'timing_std_gap', 'timing_min_gap',
            'timing_max_gap', 'timing_median_gap', 'timing_cv',
            'timing_q25', 'timing_q75', 'timing_iqr',
            'timing_burst_count', 'timing_burst_ratio',
            'timing_very_regular', 'timing_gap_uniformity',
            # New advanced timing features
            'timing_skewness', 'timing_kurtosis',
            'timing_autocorr_1', 'timing_autocorr_2',
            'timing_fft_peak_freq', 'timing_fft_peak_power', 'timing_fft_entropy',
            'timing_longest_break', 'timing_short_gap_ratio',
            'timing_gap_entropy', 'timing_benford_deviation',
            'timing_round_minute_ratio', 'timing_round_5min_ratio',
            'timing_acceleration_mean', 'timing_acceleration_std',
        ]

        if len(timestamps) >= 2:
            gaps = [(timestamps[i+1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps)-1)]
            gaps = [g for g in gaps if g >= 0]  # safety

            if gaps:
                gaps_arr = np.array(gaps)
                f['timing_mean_gap'] = np.mean(gaps_arr)
                f['timing_std_gap'] = np.std(gaps_arr)
                f['timing_min_gap'] = np.min(gaps_arr)
                f['timing_max_gap'] = np.max(gaps_arr)
                f['timing_median_gap'] = np.median(gaps_arr)
                f['timing_cv'] = np.std(gaps_arr) / max(np.mean(gaps_arr), 1)

                f['timing_q25'] = np.percentile(gaps_arr, 25)
                f['timing_q75'] = np.percentile(gaps_arr, 75)
                f['timing_iqr'] = f['timing_q75'] - f['timing_q25']

                f['timing_burst_count'] = int(np.sum(gaps_arr < 60))
                f['timing_burst_ratio'] = f['timing_burst_count'] / len(gaps)
                f['timing_very_regular'] = 1 if f['timing_cv'] < 0.5 else 0
                f['timing_gap_uniformity'] = np.std(gaps_arr) / max(np.mean(gaps_arr), 1)

                # --- ADVANCED TEMPORAL: Distributional shape ---
                n = len(gaps_arr)
                mean_g = np.mean(gaps_arr)
                std_g = max(np.std(gaps_arr), 1e-10)

                # Skewness
                f['timing_skewness'] = float(np.mean(((gaps_arr - mean_g) / std_g) ** 3)) if n >= 3 else 0
                # Kurtosis
                f['timing_kurtosis'] = float(np.mean(((gaps_arr - mean_g) / std_g) ** 4) - 3) if n >= 4 else 0

                # --- ADVANCED TEMPORAL: Autocorrelation ---
                # Lag-1 and Lag-2 autocorrelation of gaps
                # Bots with periodic posting will have high autocorrelation
                if n >= 4:
                    centered = gaps_arr - mean_g
                    var_g = np.var(gaps_arr)
                    if var_g > 0:
                        f['timing_autocorr_1'] = float(np.correlate(centered[:-1], centered[1:])[0] / ((n - 1) * var_g))
                        if n >= 5:
                            f['timing_autocorr_2'] = float(np.correlate(centered[:-2], centered[2:])[0] / ((n - 2) * var_g))
                        else:
                            f['timing_autocorr_2'] = 0
                    else:
                        f['timing_autocorr_1'] = 0
                        f['timing_autocorr_2'] = 0
                else:
                    f['timing_autocorr_1'] = 0
                    f['timing_autocorr_2'] = 0

                # --- ADVANCED TEMPORAL: FFT periodicity detection ---
                # Apply FFT to detect periodic posting patterns
                if n >= 8:
                    fft_vals = np.abs(np.fft.rfft(gaps_arr - mean_g))
                    freqs = np.fft.rfftfreq(n)
                    # Skip DC component (index 0)
                    if len(fft_vals) > 1:
                        fft_power = fft_vals[1:] ** 2
                        fft_freqs = freqs[1:]
                        peak_idx = np.argmax(fft_power)
                        f['timing_fft_peak_freq'] = float(fft_freqs[peak_idx])
                        f['timing_fft_peak_power'] = float(fft_power[peak_idx] / max(np.sum(fft_power), 1e-10))
                        # Spectral entropy (flat = high entropy = random; peaked = low = periodic)
                        fft_norm = fft_power / max(np.sum(fft_power), 1e-10)
                        fft_norm = fft_norm[fft_norm > 0]
                        f['timing_fft_entropy'] = float(-np.sum(fft_norm * np.log2(fft_norm))) if len(fft_norm) > 0 else 0
                    else:
                        f['timing_fft_peak_freq'] = 0
                        f['timing_fft_peak_power'] = 0
                        f['timing_fft_entropy'] = 0
                else:
                    f['timing_fft_peak_freq'] = 0
                    f['timing_fft_peak_power'] = 0
                    f['timing_fft_entropy'] = 0

                # --- ADVANCED TEMPORAL: Gap distribution features ---
                f['timing_longest_break'] = float(np.max(gaps_arr))
                f['timing_short_gap_ratio'] = float(np.sum(gaps_arr < 300) / n)  # < 5 min

                # Gap entropy (binned)
                if n >= 3:
                    gap_hist, _ = np.histogram(gaps_arr, bins=min(10, n))
                    gap_probs = gap_hist / max(np.sum(gap_hist), 1)
                    gap_probs = gap_probs[gap_probs > 0]
                    f['timing_gap_entropy'] = float(-np.sum(gap_probs * np.log2(gap_probs)))
                else:
                    f['timing_gap_entropy'] = 0

                # --- ADVANCED TEMPORAL: Benford's law deviation ---
                # First digits of gaps should follow Benford's law for natural data
                first_digits = [int(str(abs(int(g)))[0]) for g in gaps_arr if g > 0]
                if first_digits:
                    benford_expected = {d: math.log10(1 + 1/d) for d in range(1, 10)}
                    digit_counts = Counter(first_digits)
                    total = len(first_digits)
                    deviation = sum(abs(digit_counts.get(d, 0)/total - benford_expected[d])
                                   for d in range(1, 10))
                    f['timing_benford_deviation'] = deviation
                else:
                    f['timing_benford_deviation'] = 0

                # --- ADVANCED TEMPORAL: Round-number detection ---
                # Bots may post at suspiciously round intervals
                f['timing_round_minute_ratio'] = float(np.sum(gaps_arr % 60 < 5) / n)
                f['timing_round_5min_ratio'] = float(np.sum(gaps_arr % 300 < 15) / n)

                # --- ADVANCED TEMPORAL: Acceleration (change in posting rate) ---
                if n >= 3:
                    accels = np.diff(gaps_arr)  # second derivative of time
                    f['timing_acceleration_mean'] = float(np.mean(accels))
                    f['timing_acceleration_std'] = float(np.std(accels))
                else:
                    f['timing_acceleration_mean'] = 0
                    f['timing_acceleration_std'] = 0

            else:
                for k in TIMING_ZERO_KEYS:
                    f[k] = 0
        else:
            for k in TIMING_ZERO_KEYS:
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

            # --- NEW: Minute-level patterns ---
            minutes = [t.minute for t in timestamps]
            minute_counts = Counter(minutes)
            f['minute_entropy'] = -sum((c/len(minutes)) * np.log2(c/len(minutes))
                                       for c in minute_counts.values() if c > 0)
            f['minute_unique_count'] = len(minute_counts)

            # --- NEW: Posts per hour bins (4 time-of-day periods) ---
            morning = sum(1 for h in hours if 6 <= h < 12) / len(hours)
            afternoon = sum(1 for h in hours if 12 <= h < 18) / len(hours)
            evening = sum(1 for h in hours if 18 <= h < 24) / len(hours)
            night = sum(1 for h in hours if 0 <= h < 6) / len(hours)
            f['period_morning'] = morning
            f['period_afternoon'] = afternoon
            f['period_evening'] = evening
            f['period_night'] = night
            # Max concentration in one period
            f['period_max_concentration'] = max(morning, afternoon, evening, night)

        else:
            f['hour_unique_count'] = 0
            f['hour_entropy'] = 0
            f['hour_mode_ratio'] = 0
            f['night_post_ratio'] = 0
            f['hour_span'] = 0
            f['day_count'] = 0
            f['day_balance'] = 0
            f['minute_entropy'] = 0
            f['minute_unique_count'] = 0
            f['period_morning'] = 0
            f['period_afternoon'] = 0
            f['period_evening'] = 0
            f['period_night'] = 0
            f['period_max_concentration'] = 0

        # =====================================================================
        # 3. TEXT CONTENT FEATURES
        # =====================================================================
        TEXT_ZERO_KEYS = [
            'text_avg_length', 'text_std_length', 'text_min_length', 'text_max_length',
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
            'rt_ratio',
            # New NLP features
            'avg_word_length', 'std_word_length', 'long_word_ratio',
            'short_word_ratio', 'stopword_ratio',
            'comma_per_tweet', 'semicolon_per_tweet', 'colon_per_tweet',
            'dash_per_tweet', 'paren_per_tweet',
            'starts_with_capital_ratio', 'ends_with_punctuation_ratio',
            'avg_chars_per_word', 'function_word_ratio',
            'text_length_cv', 'word_count_cv',
            'bigram_diversity', 'trigram_diversity',
            'self_similarity_mean', 'self_similarity_std',
        ]

        if texts:
            # Basic text stats
            text_lengths = [len(t) for t in texts]
            f['text_avg_length'] = np.mean(text_lengths)
            f['text_std_length'] = np.std(text_lengths)
            f['text_min_length'] = np.min(text_lengths)
            f['text_max_length'] = np.max(text_lengths)
            f['text_length_cv'] = np.std(text_lengths) / max(np.mean(text_lengths), 1)

            # Word count per tweet
            word_counts = [len(t.split()) for t in texts]
            f['text_avg_words'] = np.mean(word_counts)
            f['text_std_words'] = np.std(word_counts)
            f['word_count_cv'] = np.std(word_counts) / max(np.mean(word_counts), 1)

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

            # --- NEW NLP: Word-level features ---
            word_lengths = [len(w) for w in all_words]
            f['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
            f['std_word_length'] = np.std(word_lengths) if word_lengths else 0
            f['long_word_ratio'] = sum(1 for wl in word_lengths if wl > 8) / max(len(word_lengths), 1)
            f['short_word_ratio'] = sum(1 for wl in word_lengths if wl <= 3) / max(len(word_lengths), 1)
            f['avg_chars_per_word'] = np.mean(word_lengths) if word_lengths else 0

            # Stopword ratio (common function words)
            en_stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                           'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                           'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                           'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                           'and', 'but', 'or', 'not', 'no', 'so', 'if', 'it', 'its',
                           'this', 'that', 'these', 'those', 'i', 'me', 'my', 'you',
                           'your', 'he', 'she', 'we', 'they', 'them', 'their'}
            fr_stopwords = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et',
                           'est', 'en', 'que', 'qui', 'dans', 'ce', 'il', 'ne', 'sur',
                           'se', 'pas', 'plus', 'par', 'je', 'avec', 'tout', 'faire',
                           'son', 'mais', 'on', 'comme', 'ou', 'si', 'leur', 'y', 'a',
                           'aux', 'au', 'nous', 'vous', 'ils', 'elle', 'elles', 'mes',
                           'ma', 'mon', 'sa', 'ses', 'tu', 'te', 'ta', 'tes'}
            stopwords = en_stopwords | fr_stopwords
            f['stopword_ratio'] = sum(1 for w in all_words if w in stopwords) / max(len(all_words), 1)
            f['function_word_ratio'] = f['stopword_ratio']  # alias

            # --- NEW NLP: Punctuation style ---
            f['comma_per_tweet'] = sum(t.count(',') for t in texts) / len(texts)
            f['semicolon_per_tweet'] = sum(t.count(';') for t in texts) / len(texts)
            f['colon_per_tweet'] = sum(t.count(':') for t in texts) / len(texts)
            f['dash_per_tweet'] = sum(t.count('-') + t.count('—') for t in texts) / len(texts)
            f['paren_per_tweet'] = sum(t.count('(') + t.count(')') for t in texts) / len(texts)

            # --- NEW NLP: Writing style patterns ---
            f['starts_with_capital_ratio'] = sum(1 for t in texts if t and t[0].isupper()) / len(texts)
            f['ends_with_punctuation_ratio'] = sum(1 for t in texts if t and t[-1] in '.!?') / len(texts)

            # --- NEW NLP: N-gram diversity ---
            all_bigrams = list(zip(all_words[:-1], all_words[1:]))
            f['bigram_diversity'] = len(set(all_bigrams)) / max(len(all_bigrams), 1)
            all_trigrams = list(zip(all_words[:-2], all_words[1:-1], all_words[2:]))
            f['trigram_diversity'] = len(set(all_trigrams)) / max(len(all_trigrams), 1)

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

            # Average sentence length
            period_count = sum(t.count('.') for t in texts)
            f['avg_sentences_per_tweet'] = period_count / len(texts)

            # Newline usage
            f['newline_per_tweet'] = sum(t.count('\n') for t in texts) / len(texts)

            # Text similarity between consecutive tweets (Jaccard)
            def jaccard(a, b):
                sa, sb = set(a.lower().split()), set(b.lower().split())
                if not sa or not sb:
                    return 0
                return len(sa & sb) / len(sa | sb)

            if len(texts) >= 2:
                consecutive_sims = [jaccard(texts[i], texts[i+1]) for i in range(len(texts)-1)]
                f['consecutive_similarity_mean'] = np.mean(consecutive_sims)
                f['consecutive_similarity_max'] = np.max(consecutive_sims)
                f['consecutive_similarity_std'] = np.std(consecutive_sims)
            else:
                f['consecutive_similarity_mean'] = 0
                f['consecutive_similarity_max'] = 0
                f['consecutive_similarity_std'] = 0

            # --- NEW NLP: Global self-similarity (all pairs sampled) ---
            if len(texts) >= 5:
                # Sample up to 50 random pairs
                import random
                random.seed(42)
                n_pairs = min(50, len(texts) * (len(texts) - 1) // 2)
                pair_sims = []
                for _ in range(n_pairs):
                    i, j = random.sample(range(len(texts)), 2)
                    pair_sims.append(jaccard(texts[i], texts[j]))
                f['self_similarity_mean'] = np.mean(pair_sims)
                f['self_similarity_std'] = np.std(pair_sims)
            else:
                f['self_similarity_mean'] = f.get('consecutive_similarity_mean', 0)
                f['self_similarity_std'] = 0

            # Topic diversity
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
            for k in TEXT_ZERO_KEYS:
                f[k] = 0

        # =====================================================================
        # 4. TF-IDF / NLP EMBEDDING FEATURES
        # =====================================================================
        idx = uid_to_idx[uid]
        for i in range(20):
            f[f'tfidf_word_{i}'] = float(tfidf_word_reduced[idx, i])
        for i in range(15):
            f[f'tfidf_char_{i}'] = float(tfidf_char_reduced[idx, i])

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


def _build_ensemble():
    """Build a fresh stacking ensemble."""
    xgb1 = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, eval_metric='logloss',
    )
    xgb2 = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.6, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=2.0, random_state=123, eval_metric='logloss',
    )
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=2,
        min_samples_split=4, random_state=42, class_weight='balanced',
    )
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=3, random_state=42,
    )
    return StackingClassifier(
        estimators=[('xgb1', xgb1), ('xgb2', xgb2), ('rf', rf), ('gb', gb)],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        cv=5, stack_method='predict_proba', passthrough=False,
    )


def select_features(X, y, feature_cols, n_target=60):
    """
    Select the best features using XGBoost importance + recursive elimination.
    Returns the indices and names of selected features.

    Strategy:
      1. Train XGBoost, rank features by importance
      2. Start with top features, incrementally add more
      3. Keep the set that maximizes CV competition score
    """
    from sklearn.model_selection import StratifiedKFold

    # Step 1: Get feature importance ranking
    xgb_selector = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss',
    )
    xgb_selector.fit(X, y)
    importances = xgb_selector.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]

    # Step 2: Forward selection - try top-K features for various K
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -999999
    best_k = 20

    # Test k = 20, 30, 40, 50, 60, 80, 100, all
    candidates = [20, 30, 40, 50, 60, 80, 100, len(feature_cols)]
    candidates = [k for k in candidates if k <= len(feature_cols)]
    candidates = sorted(set(candidates))

    print(f"  Feature selection: testing {len(candidates)} feature set sizes...")

    for k in candidates:
        selected_idx = ranked_indices[:k]
        X_sub = X[:, selected_idx]

        # Quick eval with single XGBoost (faster than full ensemble)
        xgb_quick = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss',
        )
        try:
            probas = cross_val_predict(xgb_quick, X_sub, y, cv=cv, method='predict_proba')[:, 1]
            thresh, score = find_optimal_threshold(y, probas)
            print(f"    k={k:3d} features -> score={score} (threshold={thresh:.2f})")
            if score > best_score:
                best_score = score
                best_k = k
        except:
            pass

    # Use the best K
    selected_idx = ranked_indices[:best_k]
    selected_names = [feature_cols[i] for i in selected_idx]
    print(f"  -> Selected {best_k} features (score={best_score})")

    return selected_idx, selected_names


def train_and_evaluate(train_features, train_labels, test_features=None, test_labels=None, lang="en"):
    """
    Train an ensemble model with automatic feature selection.
    """
    all_feature_cols = [c for c in train_features.columns if c != 'is_bot']

    X_train_all = train_features[all_feature_cols].values
    y_train = train_labels.values

    # Handle NaN/Inf
    X_train_all = np.nan_to_num(X_train_all, nan=0, posinf=0, neginf=0)

    print(f"\n{'='*60}")
    print(f"TRAINING ({lang.upper()}) - {len(X_train_all)} users, {sum(y_train)} bots, {len(all_feature_cols)} features")
    print(f"{'='*60}")

    # --- Feature Selection ---
    selected_idx, selected_names = select_features(X_train_all, y_train, all_feature_cols)
    feature_cols = selected_names

    X_train = X_train_all[:, selected_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Build ensemble
    ensemble = _build_ensemble()

    # --- Cross-validation ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probas = cross_val_predict(ensemble, X_train_scaled, y_train, cv=cv, method='predict_proba')[:, 1]

    best_thresh, best_cv_score = find_optimal_threshold(y_train, cv_probas)
    cv_preds = (cv_probas >= best_thresh).astype(int)
    score, tp, fn, fp, tn, max_possible = competition_score(y_train, cv_preds)

    print(f"\nCross-Validation Results (optimal threshold={best_thresh:.2f}):")
    print(f"  Competition Score: {score} / {max_possible} possible")
    print(f"  TP={tp} (bots caught), FN={fn} (bots missed), FP={fp} (humans wrongly flagged), TN={tn}")
    print(f"  Precision: {tp/max(tp+fp,1):.3f}, Recall: {tp/max(tp+fn,1):.3f}")

    # Train final model on all training data
    ensemble.fit(X_train_scaled, y_train)

    # Feature importance (from XGBoost base model)
    xgb_model = ensemble.named_estimators_['xgb1']
    importances = xgb_model.feature_importances_
    top_features = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:15]
    print(f"\nTop 15 Features (from {len(feature_cols)} selected):")
    for fname, imp in top_features:
        print(f"  {fname}: {imp:.4f}")

    # Show stacking meta-learner weights
    meta = ensemble.final_estimator_
    print(f"\nStacking meta-learner weights: {dict(zip(['xgb1','xgb2','rf','gb'], [f'{w:.3f}' for w in meta.coef_[0]]))}")

    # --- Test on held-out dataset if provided ---
    if test_features is not None and test_labels is not None:
        X_test_all = test_features[all_feature_cols].values
        y_test = test_labels.values
        X_test_all = np.nan_to_num(X_test_all, nan=0, posinf=0, neginf=0)
        X_test = X_test_all[:, selected_idx]
        X_test_scaled = scaler.transform(X_test)

        test_probas = ensemble.predict_proba(X_test_scaled)[:, 1]

        test_preds = (test_probas >= best_thresh).astype(int)
        score, tp, fn, fp, tn, max_possible = competition_score(y_test, test_preds)

        print(f"\nHeld-Out Test Results (threshold={best_thresh:.2f}):")
        print(f"  Competition Score: {score} / {max_possible} possible")
        print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        print(f"  Precision: {tp/max(tp+fp,1):.3f}, Recall: {tp/max(tp+fn,1):.3f}")

        best_test_thresh, best_test_score = find_optimal_threshold(y_test, test_probas)
        test_preds_opt = (test_probas >= best_test_thresh).astype(int)
        score_opt, tp2, fn2, fp2, tn2, _ = competition_score(y_test, test_preds_opt)
        print(f"\n  (Oracle test threshold={best_test_thresh:.2f} would give score={score_opt})")

    # Return selected_idx too so detection mode can use same features
    return ensemble, scaler, best_thresh, feature_cols, selected_idx, all_feature_cols


def predict_bots(model, scaler, threshold, features_df, all_feature_cols, selected_idx):
    """Predict bot IDs from a new dataset."""
    X_all = features_df[all_feature_cols].values
    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)
    X = X_all[:, selected_idx]
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
        selected_idx = saved['selected_idx']
        all_feature_cols = saved['all_feature_cols']

        bot_ids, probas = predict_bots(model, scaler, threshold, features, all_feature_cols, selected_idx)

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
        *_, = train_and_evaluate(
            datasets[30]['features'], datasets[30]['labels'],
            datasets[32]['features'], datasets[32]['labels'],
            lang='en'
        )

        # Train on 32, test on 30
        print("\n--- Train on Dataset 32, Test on Dataset 30 ---")
        *_, = train_and_evaluate(
            datasets[32]['features'], datasets[32]['labels'],
            datasets[30]['features'], datasets[30]['labels'],
            lang='en'
        )

        # Train on BOTH for final model
        print("\n--- Final English Model: Train on 30+32 ---")
        en_features = pd.concat([datasets[30]['features'], datasets[32]['features']])
        en_labels = pd.concat([datasets[30]['labels'], datasets[32]['labels']])
        model_en, scaler_en, thresh_en, fcols_en, selidx_en, allfcols_en = train_and_evaluate(
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
        *_, = train_and_evaluate(
            datasets[31]['features'], datasets[31]['labels'],
            datasets[33]['features'], datasets[33]['labels'],
            lang='fr'
        )

        # Train on 33, test on 31
        print("\n--- Train on Dataset 33, Test on Dataset 31 ---")
        *_, = train_and_evaluate(
            datasets[33]['features'], datasets[33]['labels'],
            datasets[31]['features'], datasets[31]['labels'],
            lang='fr'
        )

        # Train on BOTH for final model
        print("\n--- Final French Model: Train on 31+33 ---")
        fr_features = pd.concat([datasets[31]['features'], datasets[33]['features']])
        fr_labels = pd.concat([datasets[31]['labels'], datasets[33]['labels']])
        model_fr, scaler_fr, thresh_fr, fcols_fr, selidx_fr, allfcols_fr = train_and_evaluate(
            fr_features, fr_labels, lang='fr'
        )

        # =====================================================================
        # SAVE MODELS
        # =====================================================================
        import pickle
        os.makedirs('models', exist_ok=True)

        for lang_name, model, scaler, thresh, fcols, selidx, allfcols in [
            ('en', model_en, scaler_en, thresh_en, fcols_en, selidx_en, allfcols_en),
            ('fr', model_fr, scaler_fr, thresh_fr, fcols_fr, selidx_fr, allfcols_fr),
        ]:
            model_path = os.path.join('models', f'model_{lang_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'threshold': thresh,
                    'feature_cols': fcols,
                    'selected_idx': selidx,
                    'all_feature_cols': allfcols,
                }, f)
            lang = lang_name  # for the print statement
            print(f"\nSaved {lang} model to {model_path} (threshold={thresh:.2f})")

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print("\nTo detect bots on a new dataset:")
        print("  python3 bot_detector.py detect <dataset.json>")
