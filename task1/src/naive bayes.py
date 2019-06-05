from collections import Counter
import classifier.py

HANDLES_DICT = {0: "@realDonaldTrump",
                1: "@joebiden",
                2: "@ConanOBrien",
                3: "@TheEllenShow",
                4: "@KimKardashian",
                5: "@KingJames",
                6: "@ladygaga",
                7: "@Cristiano",
                8: "@jimmykimmel",
                9: "@Schwartzenegger",
                }


class naive_bayes(classifier)


def compute_probabilities(words, people, texts):

    probabilitie = Counter()

    for word in words:
        for person in people:
            num_words =
            probabilitie[(word, person)] = 1


def add_one_smoothing(probs, words, people):
    smoothed_probs = Counter()
    for word in words:
        for person in people:
            smoothed_probs[(word, person)] = (probs[(word, person)] + 1) / (len(words) + 1)

    return smoothed_probs


def classify_tweet(probs, sentence, people):

    results = []
    for person in people:
        result = 1
        for word in sentence:
            result *= probs[(word, person)]
        results.append(result)

