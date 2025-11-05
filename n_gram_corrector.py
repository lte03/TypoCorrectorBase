import Levenshtein
from typing import Dict, Tuple, Optional
import nltk


def load_ngram_frequencies(filepath: str) -> Dict:
    ngram_freq = {}
    
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            ngram, freq = line.rsplit('\t', 1)
            freq = int(freq.strip())
            words = ngram.split()
            ngram_freq[tuple(words) if len(words) > 1 else words[0]] = freq
    
    return ngram_freq


def context_aware_word_correction(
    misspelled_word: str,
    prev_word: Optional[str],
    next_word: Optional[str],
    prev_prev_word: Optional[str],
    next_next_word: Optional[str],
    unigram_freq: Dict[str, int],
    bigram_freq: Dict[Tuple[str, str], int],
    trigram_freq: Dict[Tuple[str, str, str], int]
) -> str:
    def similarity_score(word1: str, word2: str) -> float:
        if abs(len(word1) - len(word2)) > 2:
            return 0
        return sum(1 for a, b in zip(word1, word2) if a == b) / max(len(word1), len(word2))

    possible_corrections = sorted(
        unigram_freq.keys(),
        key=lambda word: Levenshtein.distance(misspelled_word, word)
    )[:50]

    best_word = None
    highest_probability = 0

    for correction in possible_corrections:
        bigram_probability = 1.0
        trigram_probability = 1.0

        if prev_word:
            prev_bigram = (prev_word, correction)
            if prev_bigram in bigram_freq:
                bigram_probability *= bigram_freq[prev_bigram] / unigram_freq.get(prev_word, 1)
            else:
                bigram_probability *= 0.0001

        if next_word:
            next_bigram = (correction, next_word)
            if next_bigram in bigram_freq:
                bigram_probability *= bigram_freq[next_bigram] / unigram_freq.get(correction, 1)
            else:
                bigram_probability *= 0.0001

        if prev_word and prev_prev_word:
            prev_trigram = (prev_prev_word, prev_word, correction)
            if prev_trigram in trigram_freq:
                trigram_probability *= (
                    trigram_freq[prev_trigram] /
                    bigram_freq.get((prev_prev_word, prev_word), 1)
                )
            else:
                trigram_probability *= 0.00001

        if next_word and next_next_word:
            next_trigram = (correction, next_word, next_next_word)
            if next_trigram in trigram_freq:
                trigram_probability *= (
                    trigram_freq[next_trigram] /
                    bigram_freq.get((correction, next_word), 1)
                )
            else:
                trigram_probability *= 0.00001

        total_probability = (
            bigram_probability * 0.4 +
            trigram_probability * 0.4 +
            similarity_score(misspelled_word, correction) * 0.2
        )

        if total_probability > highest_probability:
            highest_probability = total_probability
            best_word = correction

    return best_word if best_word else misspelled_word


def correct_sentence(sentence: str, unigram_freq: Dict[str, int], 
                    bigram_freq: Dict[Tuple[str, str], int],
                    trigram_freq: Dict[Tuple[str, str, str], int]) -> str:

    tokens = nltk.word_tokenize(sentence.lower())
    corrected_tokens = tokens.copy()
    
    for i in range(len(tokens)):
        prev_prev_word = tokens[i-2] if i >= 2 else None
        prev_word = tokens[i-1] if i >= 1 else None
        next_word = tokens[i+1] if i < len(tokens)-1 else None
        next_next_word = tokens[i+2] if i < len(tokens)-2 else None
        
        corrected_word = context_aware_word_correction(
            tokens[i],
            prev_word,
            next_word,
            prev_prev_word,
            next_next_word,
            unigram_freq,
            bigram_freq,
            trigram_freq
        )
        
        if corrected_word:
            corrected_tokens[i] = corrected_word
    
    return ' '.join(corrected_tokens)
