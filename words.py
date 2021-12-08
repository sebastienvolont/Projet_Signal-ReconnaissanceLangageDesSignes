import levenshtein


class Dictionnary:
    def __init__(self, filename: str):
        with open(filename, 'r') as f:
            self.words = set(f.read().splitlines())

    def get_alike_words(self, word: str, threshold: float = 75) -> list[str]:
        return [w for w in self.words if levenshtein.ratio(word, w) > threshold]

    def get_nearest_word(self, word: str) -> str:
        if word in self.words:
            return word

        return min(self.words, key=lambda w: levenshtein.distance(word, w))
