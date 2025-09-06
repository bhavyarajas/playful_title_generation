# Detecting Playful/Funny Paper Titles

## Goal: classify paper titles by how "humourous" they are without existing labeled data. I computed three features and combined them to a single humor score (0–1).

## Features
	
### Scientific alignment (using SciBERT)
	•	I use the SciBERT model to estimate how “expected” the words in a title are in scientific/academic writing.
	•	My understanding: this model tokenizes the title, then masks a few tokens one at a time (up to 5 inner tokens). For each mask, SciBERT predicts the missing token from its surrounding context. If SciBERT assigns a high log-probability to the actual true token, then that word “fits” scientific style. I averaged these log-probabilities into a variable called sci_avg_logp.
	•	fundamental idea: lower sci_avg_logp -> the title feels less like typical scientific language -> therefore more likely playful.

### Punctuation/Special Character score
	•	I wrote functions to count instances of ! ? : ; quotes, parentheses, dashes, and ellipses.
    •	This is squashed to 0–1 with a sigmoid function, later stored in punct_store

### Sentiment Analysis (using VADER)
	•	My understanding: VADER returns a sentiment score between [−1, 1]. I rescaled it to [0, 1] (WHY) and stored it in sent_pos.
    •	It's might not be the most important feature but it tends to be higher for upbeat/cheeky titles.

## Scoring + Feature Combinationa

I z-scored each feature (sci_avg_logp, punct_score, sent_pos) to put them on a comparable scale, then computed a linear combination and passed it through a sigmoid to get humor_score between [0,1]:
	•	Weights were chosen for sensible behavior after quick sanity checks:
	•	weight for sci_avg_logp: −1.0 (negative sign because lower plausibility should increase humor)
	•	weight for punct_score: +0.8
	•	weight for sent_pos: +0.3

I also create the option of choosing between:
	•	A K-Means on the features.x
	•	A threshold classification. 


## How to Make it Better

There are a number of straightforward (and some creative) ways I could improve this classification if I had more time and resources:
- Quantitative evaluation of K-Means vs. threshold: Right now, I offered both options without choosing one. With a small labeled validation set, I could compare the two using F1 scores and ROC/AUC analysis. This would tell me which method separates playful vs. boring titles better.
- Creating other signals: Your examples showed the use of font size, bolding, or underlining in titles. Incorporating these as additional features could capture more playfulness cues.
- Better punctuation analysis: Instead of just counting punctuation, I could actually test how much punctuation correlates with humor in the dataset. For example, is ? or repeated dashes strongly predictive? If not, weights could be adjusted (or even learned) instead of assigned heuristically.
- Model ensemble (SciBERT + RoBERTa): Currently I only used SciBERT for “scientific alignment.” Pairing it with a general language model like RoBERTa would let me measure the difference between “scientific" and “general” tokenization. That gap could be a stronger indicator of playfulness.


Note: Due to time and compute limits, I kept the classification simple and only ran on the first 1,000 rows. With more time, I would incorporate richer features, run proper evaluation metrics, and test SciBERT + RoBERTa together for more robust results.
