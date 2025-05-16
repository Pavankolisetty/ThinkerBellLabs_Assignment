import re
from collections import defaultdict
import heapq
import sys

# Unbuffered I/O for interactive mode
sys.stdout.reconfigure(line_buffering=True)
sys.stdin.reconfigure(line_buffering=True)

DOT_TO_KEY = {1: 'D', 2: 'W', 3: 'Q', 4: 'K', 5: 'O', 6: 'P'}

ENGLISH_BRAILLE = {
    'a': [1], 'b': [1, 2], 'c': [1, 4], 'd': [1, 4, 5], 'e': [1, 5],
    'f': [1, 2, 4], 'g': [1, 2, 4, 5], 'h': [1, 2, 5], 'i': [2, 4],
    'j': [2, 4, 5], 'k': [1, 3], 'l': [1, 2, 3], 'm': [1, 3, 4],
    'n': [1, 3, 4, 5], 'o': [1, 3, 5], 'p': [1, 2, 3, 4], 'q': [1, 2, 3, 4, 5],
    'r': [1, 2, 3, 5], 's': [2, 3, 4], 't': [2, 3, 4, 5], 'u': [1, 3, 6],
    'v': [1, 2, 3, 6], 'w': [2, 4, 5, 6], 'x': [1, 3, 4, 6], 'y': [1, 3, 4, 5, 6],
    'z': [1, 3, 5, 6]
}

FRENCH_BRAILLE = ENGLISH_BRAILLE.copy()

CONTRACTIONS = {
    'the': [2, 3, 4, 6]
}

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None

class BrailleAutocorrect:
    def __init__(self):
        self.dictionaries = defaultdict(TrieNode)
        self.corrections = defaultdict(int)
        self.braille_maps = {
            'english': ENGLISH_BRAILLE,
            'french': FRENCH_BRAILLE
        }
        self.contractions = CONTRACTIONS

    def add_word(self, word, language='english'):
        node = self.dictionaries[language]
        for char in word.lower():
            dots = tuple(self.braille_maps[language].get(char, []))
            if dots not in node.children:
                node.children[dots] = TrieNode()
            node = node.children[dots]
        node.is_end = True
        node.word = word

    def load_dictionary(self, words, language='english'):
        for word in words:
            self.add_word(word, language)
        if language == 'english':
            for word, dots in self.contractions.items():
                node = self.dictionaries[language]
                if tuple(dots) not in node.children:
                    node.children[tuple(dots)] = TrieNode()
                node.children[tuple(dots)].is_end = True
                node.children[tuple(dots)].word = word

    def keys_to_dots(self, keys):
        if not keys:
            return []
        key_set = set(keys.split('+'))
        valid_keys = set(DOT_TO_KEY.values())
        if not key_set.issubset(valid_keys):
            raise ValueError(f"Invalid keys detected: {key_set - valid_keys}. Valid keys are {valid_keys}.")
        return sorted(dot for dot, key in DOT_TO_KEY.items() if key in key_set)

    def input_to_braille(self, input_seq):
        return [self.keys_to_dots(keys) for keys in input_seq.split()]

    def hamming_distance(self, dots1, dots2):
        set1, set2 = set(dots1), set(dots2)
        return len(set1.symmetric_difference(set2))

    def normalized_dot_distance(self, dots1, dots2):
        """Smarter similarity: normalized distance based on overlap."""
        set1, set2 = set(dots1), set(dots2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        if union == 0:
            return 1.0
        return 1 - (intersection / union)

    def levenshtein_distance(self, input_dots, word_dots):
        len_diff_penalty = abs(len(input_dots) - len(word_dots)) * 0.5
        if len(input_dots) < len(word_dots):
            input_dots, word_dots = word_dots, input_dots
        if not word_dots:
            return len(input_dots) + len_diff_penalty

        previous_row = range(len(word_dots) + 1)
        for i, c1 in enumerate(input_dots):
            current_row = [i + 1]
            for j, c2 in enumerate(word_dots):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitution_cost = self.normalized_dot_distance(c1, c2)
                substitutions = previous_row[j] + substitution_cost
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1] + len_diff_penalty

    def suggest_word(self, input_seq, language='english', max_suggestions=1):
        input_dots = self.input_to_braille(input_seq)
        if not input_dots:
            return []
        suggestions = []
        def dfs(node, current_dots, word_path):
            if node.is_end and node.word:
                dist = self.levenshtein_distance(input_dots, current_dots)
                learn_bonus = min(0.5, 0.1 * self.corrections[node.word.lower()])
                score = dist - learn_bonus
                heapq.heappush(suggestions, (score, dist, node.word))
            for dots, child in node.children.items():
                dfs(child, current_dots + [list(dots)], word_path + [dots])
        dfs(self.dictionaries[language], [], [])
        suggestions.sort()
        return [word for _, _, word in suggestions[:max_suggestions]]

    def learn_correction(self, input_seq, corrected_word):
        self.corrections[corrected_word.lower()] += 1

    def process_input(self, input_seq, language='english'):
        return self.suggest_word(input_seq, language)

def run_tests():
    autocorrect = BrailleAutocorrect()
    english_words = ['cat', 'hat', 'the', 'hello', 'dog', 'bad']
    french_words = ['chat', 'ami', 'amie']
    autocorrect.load_dictionary(english_words, 'english')
    autocorrect.load_dictionary(french_words, 'french')

    test_cases = [
        {'input': 'D+K D W+Q+O', 'language': 'english', 'expected': ['cat']},
        {'input': 'D+K+O D W+Q+O', 'language': 'english', 'expected': ['cat']},
        {'input': 'D D W+Q+O', 'language': 'english', 'expected': ['cat']},
        {'input': 'D+K D+K+O D W+Q+O', 'language': 'french', 'expected': ['chat']},
        {'input': 'W+Q+K+P', 'language': 'english', 'expected': ['the']},
        {'input': 'D W+Q+O', 'language': 'english', 'expected': ['cat']},
        {'input': 'D+W+O D+O D+W+Q D+W+Q D+Q+O', 'language': 'english', 'expected': ['hello']},
        {'input': 'W+K D W+Q+O', 'language': 'english', 'expected': ['cat']},
        {'input': '', 'language': 'english', 'expected': []},
        {'input': 'W+K+O Q+K W+Q+O', 'language': 'english', 'expected': ['dog']},
        {'input': 'D D+O W+K D', 'language': 'french', 'expected': ['amie']},
        {'input': 'D+K D W+Q+O W+Q+K+P', 'language': 'english', 'expected': ['cat']},
        {'input': 'A+B C+D', 'language': 'english', 'expected': 'error'}
    ]

    for i, test in enumerate(test_cases, 1):
        autocorrect.corrections.clear()
        try:
            result = autocorrect.process_input(test['input'], test['language'])
            print(f"Test {i}: Input: {test['input']} | Language: {test['language']}")
            print(f"Expected: {test['expected']} | Got: {result}")
            print("Pass" if result == test['expected'] else "Fail")
        except ValueError as e:
            if test['expected'] == 'error':
                print(f"Test {i}: Expected ValueError | Got: {str(e)}")
                print("Pass")
            else:
                print(f"Test {i}: Unexpected Error - {str(e)}")
                print("Fail")

if __name__ == "__main__":
    print("Running all tests...")
    run_tests()
