import re
from collections import defaultdict
import heapq
import sys

# Ensure unbuffered input/output for interactive testing
sys.stdout.reconfigure(line_buffering=True)
sys.stdin.reconfigure(line_buffering=True)

# Braille dot-to-qwert key mapping
DOT_TO_KEY = {1: 'D', 2: 'W', 3: 'Q', 4: 'K', 5: 'O', 6: 'P'}

# English Braille mappings (letter to dot pattern)
ENGLISH_BRAILLE = {
    'a': [1], 'b': [1, 2], 'c': [1, 4], 'd': [1, 4, 5], 'e': [1, 5],
    'f': [1, 2, 4], 'g': [1, 2, 4, 5], 'h': [1, 2, 5], 'i': [2, 4],
    'j': [2, 4, 5], 'k': [1, 3], 'l': [1, 2, 3], 'm': [1, 3, 4],
    'n': [1, 3, 4, 5], 'o': [1, 3, 5], 'p': [1, 2, 3, 4], 'q': [1, 2, 3, 4, 5],
    'r': [1, 2, 3, 5], 's': [2, 3, 4], 't': [2, 3, 4, 5], 'u': [1, 3, 6],
    'v': [1, 2, 3, 6], 'w': [2, 4, 5, 6], 'x': [1, 3, 4, 6], 'y': [1, 3, 4, 5, 6],
    'z': [1, 3, 5, 6]
}

# French Braille mappings (complete A-Z)
FRENCH_BRAILLE = {
    'a': [1], 'b': [1, 2], 'c': [1, 4], 'd': [1, 4, 5], 'e': [1, 5],
    'f': [1, 2, 4], 'g': [1, 2, 4, 5], 'h': [1, 2, 5], 'i': [2, 4],
    'j': [2, 4, 5], 'k': [1, 3], 'l': [1, 2, 3], 'm': [1, 3, 4],
    'n': [1, 3, 4, 5], 'o': [1, 3, 5], 'p': [1, 2, 3, 4], 'q': [1, 2, 3, 4, 5],
    'r': [1, 2, 3, 5], 's': [2, 3, 4], 't': [2, 3, 4, 5], 'u': [1, 3, 6],
    'v': [1, 2, 3, 6], 'w': [2, 4, 5, 6], 'x': [1, 3, 4, 6], 'y': [1, 3, 4, 5, 6],
    'z': [1, 3, 5, 6]
}

# Braille contractions (English only, using standard Braille)
CONTRACTIONS = {
    'the': [2, 3, 4, 6]  # Standard English Braille contraction for "the"
}

# Trie node for dictionary storage
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
        """Add a word to the Trie for the specified language."""
        node = self.dictionaries[language]
        for char in word.lower():
            dots = tuple(self.braille_maps[language].get(char, []))
            if dots not in node.children:
                node.children[dots] = TrieNode()
            node = node.children[dots]
        node.is_end = True
        node.word = word

    def load_dictionary(self, words, language='english'):
        """Load a list of words into the dictionary."""
        for word in words:
            self.add_word(word, language)
        # Add contractions for English only
        if language == 'english':
            for word, dots in self.contractions.items():
                node = self.dictionaries[language]
                if tuple(dots) not in node.children:
                    node.children[tuple(dots)] = TrieNode()
                node.children[tuple(dots)].is_end = True
                node.children[tuple(dots)].word = word

    def keys_to_dots(self, keys):
        """Convert QWERTY key combination (e.g., 'D+K') to dot pattern."""
        if not keys:
            return []
        key_set = set(keys.split('+'))
        # Validate keys
        valid_keys = set(DOT_TO_KEY.values())
        if not key_set.issubset(valid_keys):
            raise ValueError(f"Invalid keys detected: {key_set - valid_keys}. Valid keys are {valid_keys}.")
        dots = []
        for dot, key in DOT_TO_KEY.items():
            if key in key_set:
                dots.append(dot)
        return sorted(dots)

    def input_to_braille(self, input_seq):
        """Convert input sequence (e.g., 'D+K D W+Q+O') to list of dot patterns."""
        return [self.keys_to_dots(keys) for keys in input_seq.split()]

    def hamming_distance(self, dots1, dots2):
        """Calculate Hamming distance between two dot patterns."""
        set1, set2 = set(dots1), set(dots2)
        return len(set1.symmetric_difference(set2))

    def levenshtein_distance(self, input_dots, word_dots):
        """Calculate Levenshtein distance between two sequences of dot patterns."""
        length_diff = abs(len(input_dots) - len(word_dots)) * 0.5
        if len(input_dots) < len(word_dots):
            input_dots, word_dots = word_dots, input_dots
        if not word_dots:
            return len(input_dots) + length_diff
        previous_row = range(len(word_dots) + 1)
        for i, c1 in enumerate(input_dots):
            current_row = [i + 1]
            for j, c2 in enumerate(word_dots):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                # Scale substitution cost by Hamming distance (normalized)
                hamming = self.hamming_distance(c1, c2)
                substitutions = previous_row[j] + (hamming / 6 if hamming > 0 else 0)  
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1] + length_diff

    def suggest_word(self, input_seq, language='english', max_suggestions=1):
        """Suggest the closest word(s) for the input sequence."""
        input_dots = self.input_to_braille(input_seq)
        # Handle empty input
        if not input_dots:
            return []
        suggestions = []
        def search_trie(node, current_dots, word_chars):
            if node.is_end and node.word:
                distance = self.levenshtein_distance(input_dots, current_dots)
                length_adjustment = abs(len(current_dots) - len(input_dots)) * 0.5
                learning_adjustment = min(0.5, 0.1 * self.corrections[node.word])
                score = distance + length_adjustment - learning_adjustment
                # Store raw distance for tie-breaking
                heapq.heappush(suggestions, (score, distance, node.word))
            for dots, child in node.children.items():
                search_trie(child, current_dots + [dots], word_chars + [dots])

        search_trie(self.dictionaries[language], [], [])
        suggestions = sorted(suggestions, key=lambda x: (x[0], x[1], x[2]))
        return [word for _, _, word in suggestions[:max_suggestions]]

    def learn_correction(self, input_seq, corrected_word):
        """Update correction frequency for learning mechanism."""
        self.corrections[corrected_word.lower()] += 1

    def process_input(self, input_seq, language='english'):
        """Process input and return suggestions."""
        return self.suggest_word(input_seq, language)

# Example usage and test cases
def run_tests():
    autocorrect = BrailleAutocorrect()
    # Load sample dictionary
    english_words = ['cat', 'hat', 'the', 'hello', 'dog', 'bad']
    french_words = ['chat', 'ami', 'amie']
    autocorrect.load_dictionary(english_words, 'english')
    autocorrect.load_dictionary(french_words, 'french')

    # Test cases
    test_cases = [
        # Test 1: Exact match
        {
            'input': 'D+K D W+Q+O',  # 'cat' in English Braille
            'language': 'english',
            'expected': ['cat']
        },
        # Test 2: Error - Extra dot
        {
            'input': 'D+K+O D W+Q+O',  # 'cat' with extra dot 5 in 'c'
            'language': 'english',
            'expected': ['cat']
        },
        # Test 3: Error - Missing dot
        {
            'input': 'D D W+Q+O',  # 'cat' with missing dot 4 in 'c'
            'language': 'english',
            'expected': ['cat']
        },
        # Test 4: French dictionary
        {
            'input': 'D+K D+K+O D W+Q+O',  # 'chat' in French Braille
            'language': 'french',
            'expected': ['chat']
        },
        # Test 5: Contraction
        {
            'input': 'W+Q+K+P',  # 'the' contraction (standard)
            'language': 'english',
            'expected': ['the']
        },
        # Test 6: Word not in dictionary (should suggest closest match)
        {
            'input': 'D W+Q+O',  # 'bat' (not in dictionary), closest to 'cat'
            'language': 'english',
            'expected': ['cat']
        },
        # Test 7: Longer word (test performance)
        {
            'input': 'D+W+O D+O D+W+Q D+W+Q D+Q+O',  # 'hello' in English Braille
            'language': 'english',
            'expected': ['hello']
        },
        # Test 8: Mistyped dot (QWERTY adjacency, W instead of D)
        {
            'input': 'W+K D W+Q+O',  # Intended 'cat', but 'c' mistyped (W+K instead of D+K)
            'language': 'english',
            'expected': ['cat']
        },
        # Test 9: Empty input (edge case)
        {
            'input': '',  # Empty input
            'language': 'english',
            'expected': []
        },
        # Test 10: New word with slight error
        {
            'input': 'W+K+O Q+K W+Q+O',  # 'dog' with mistyped 'd' (W+K+O instead of D+K+O)
            'language': 'english',
            'expected': ['dog']
        },
        # Test 11: Longer French word
        {
            'input': 'D D+O W+K D',  # 'amie' in French Braille
            'language': 'french',
            'expected': ['amie']
        },
        # Test 12: Multiple words in input
        {
            'input': 'D+K D W+Q+O W+Q+K+P',  # 'cat the' in English Braille
            'language': 'english',
            'expected': ['cat']
        },
        # Test 13: Completely invalid input
        {
            'input': 'A+B C+D',  # Invalid keys A, B, C
            'language': 'english',
            'expected': 'error'  # Expecting ValueError
        }
    ]

    for i, test in enumerate(test_cases, 1):
        # Reset corrections to avoid test interference
        autocorrect.corrections.clear()
        try:
            result = autocorrect.process_input(test['input'], test['language'])
            print(f"Test {i}: Input: {test['input']}, Language: {test['language']}")
            print(f"Expected: {test['expected']}, Got: {result}")
            print("Pass" if result == test['expected'] else "Fail")
            # Simulate learning
            if result:
                autocorrect.learn_correction(test['input'], result[0])
        except ValueError as e:
            print(f"Test {i}: Input: {test['input']}, Language: {test['language']}")
            if test['expected'] == 'error':
                print(f"Expected: ValueError, Got: {e}")
                print("Pass")
            else:
                print(f"Unexpected error: {e}")
                print("Fail")

# Interactive testing mode
def interactive_test():
    autocorrect = BrailleAutocorrect()
    # Load sample dictionary
    english_words = ['cat', 'hat', 'the', 'hello', 'dog', 'bad']
    french_words = ['chat', 'ami', 'amie']
    autocorrect.load_dictionary(english_words, 'english')
    autocorrect.load_dictionary(french_words, 'french')

    print("\nInteractive Braille Autocorrect Testing")
    print("Enter QWERTY key combinations (e.g., 'D+K D W+Q+O') to get suggestions.")
    print("Valid keys: D, W, Q, K, O, P (separated by '+' for simultaneous presses).")
    print("Type 'exit' to quit.")
    print("Languages: 'english' or 'french'.")

    while True:
        # Reset corrections at the start of each iteration to avoid bias
        autocorrect.corrections.clear()
        try:
            language = input("Enter language (english/french): ").strip().lower()
            if language not in ['english', 'french']:
                print("Invalid language. Please enter 'english' or 'french'.")
                continue

            user_input = input("Enter Braille input: ").strip()
            if user_input.lower() == 'exit':
                break

            result = autocorrect.process_input(user_input, language)
            print(f"Suggestion: {result}")
            if result:
                autocorrect.learn_correction(user_input, result[0])
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    print("Running predefined test cases...")
    run_tests()
    print("\nStarting interactive testing...")
    interactive_test()