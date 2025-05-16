import streamlit as st
from corelogic import BrailleAutocorrect

# Initialize autocorrect system
autocorrect = BrailleAutocorrect()

# Load dictionary
english_words = ['cat', 'hat', 'the', 'hello', 'dog', 'bad']
french_words = ['chat', 'ami', 'amie']
autocorrect.load_dictionary(english_words, 'english')
autocorrect.load_dictionary(french_words, 'french')

st.title("🔠 Braille Autocorrect & Suggestion System")
st.markdown("Enter Braille using QWERTY keys (e.g., `D+K D W+Q+O`). Each character = one key group.")

st.subheader("🧪 Sample Test Cases")

# Test cases from your `run_tests()` function
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

results = []
passed = 0

for test in test_cases:
    autocorrect.corrections.clear()
    try:
        actual = autocorrect.process_input(test['input'], test['language'])
        pass_status = actual == test['expected']
    except ValueError:
        actual = 'error'
        pass_status = test['expected'] == 'error'

    if pass_status:
        passed += 1

    results.append({
        'Input': test['input'],
        'Lang': test['language'],
        'Expected': str(test['expected']),
        'Got': str(actual),
        'Pass': '✅' if pass_status else '❌'
    })

# Display results table
st.dataframe(results, use_container_width=True)

# Show accuracy
accuracy = round((passed / len(test_cases)) * 100, 2)
st.success(f"🧠 System Accuracy: {accuracy}% ({passed}/{len(test_cases)} passed)")

st.divider()

# User interaction section
st.subheader("💬 Try It Yourself")

language = st.selectbox("Select Language:", ['english', 'french'])
braille_input = st.text_input("Enter Braille QWERTY Input (e.g., `D+K D W+Q+O`):")

if braille_input:
    try:
        suggestions = autocorrect.process_input(braille_input.strip(), language)
        if suggestions:
            st.success(f"Suggested Word: **{suggestions[0]}**")
            autocorrect.learn_correction(braille_input.strip(), suggestions[0])
        else:
            st.warning("No suggestions found.")
    except ValueError as e:
        st.error(f"Invalid input: {e}")
