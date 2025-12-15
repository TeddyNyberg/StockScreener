from collections import defaultdict
import pickle
import os
import re

def get_shape_compressed(word):
    shape = []
    for char in word:
        if char.isupper():
            symbol = "X"
        elif char.islower():
            symbol = "x"
        elif char.isdigit():
            symbol = "d"
        else:
            symbol = char
        if not shape or shape[-1] != symbol:
            shape.append(symbol)
    return "".join(shape)


def get_prob(count, total):
    return (count + 1) / (total + 1e5)

def dd_int():
    return defaultdict(int)

class InteractiveNER:
    def __init__(self):
        self.transitions = defaultdict(dd_int)
        self.emissions = defaultdict(dd_int)
        self.word_counts = defaultdict(int)
        self.shape_counts = defaultdict(int)
        self.known_vocab = set()

    def save(self, filename="ner_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"--> Model saved to {filename}")

    @staticmethod
    def load(filename="ner_model.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                model = pickle.load(f)
            print(f"Loaded existing model from {filename}")
            return model
        else:
            print("No saved model found. Creating new one.")
            return InteractiveNER()

    def train(self, training_data, weight=1):
        for sentence in training_data:
            prev_tag = "START"
            self.word_counts["START"] += weight

            for word, tag in sentence:
                self.emissions[tag][word.lower()] += weight
                self.known_vocab.add(word.lower())

                shape_token = "SHAPE:" + get_shape_compressed(word)
                self.emissions[tag][shape_token] += weight

                self.transitions[prev_tag][tag] += weight
                self.word_counts[tag] += weight
                self.shape_counts[tag] += weight

                prev_tag = tag

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text)

    def predict(self, text):
        tokens = self.tokenize(text)
        if not tokens:
            return [], []
        states = list(self.word_counts.keys())
        if "START" in states:
            states.remove("START")

        scores = []
        first_scores = {}

        first_word = tokens[0]
        first_shape = "SHAPE:" + get_shape_compressed(first_word)

        for tag in states:
            trans_p = get_prob(self.transitions["START"][tag], self.word_counts["START"])

            if first_word.lower() in self.known_vocab:
                emit_p = get_prob(self.emissions[tag][first_word.lower()], self.word_counts[tag])
            else:
                emit_p = get_prob(self.emissions[tag][first_shape], self.shape_counts[tag])

            first_scores[tag] = (trans_p * emit_p, "START")
        scores.append(first_scores)

        for i in range(1, len(tokens)):
            curr_word = tokens[i]
            curr_shape = "SHAPE:" + get_shape_compressed(curr_word)
            curr_scores = {}

            for curr_tag in states:
                best_prev_score = -1
                best_prev_tag = None

                if curr_word.lower() in self.known_vocab:
                    emit_p = get_prob(self.emissions[curr_tag][curr_word.lower()], self.word_counts[curr_tag])
                else:
                    emit_p = get_prob(self.emissions[curr_tag][curr_shape], self.shape_counts[curr_tag])

                for prev_tag in states:
                    prev_score = scores[i - 1][prev_tag][0]
                    trans_p = get_prob(self.transitions[prev_tag][curr_tag], self.word_counts[prev_tag])

                    score = prev_score * trans_p * emit_p

                    if score > best_prev_score:
                        best_prev_score = score
                        best_prev_tag = prev_tag
                curr_scores[curr_tag] = (best_prev_score, best_prev_tag)
            scores.append(curr_scores)

        best_end_score = -1
        best_end_tag = None
        last_idx = len(tokens) - 1
        for tag, (score, _) in scores[last_idx].items():
            if score > best_end_score:
                best_end_score = score
                best_end_tag = tag

        final_tags = [best_end_tag]
        curr_tag = best_end_tag
        for i in range(last_idx, 0, -1):
            _, prev_tag = scores[i][curr_tag]
            final_tags.insert(0, prev_tag)
            curr_tag = prev_tag

        return tokens, final_tags


def interactive_train():
    model = InteractiveNER.load("ner_model.pkl")

    inspect_model(model)

    if len(model.known_vocab) == 0:
        print("--> Model is new. Training on baseline data...")
        train_data = [
            [("I", "O"), ("live", "O"), ("in", "O"), ("New", "B-LOC"), ("York", "I-LOC")],
            [("I", "O"), ("flew", "O"), ("to", "O"), ("Paris", "B-LOC")],
            [("Steve", "B-PER"), ("Jobs", "I-PER"), ("founded", "O"), ("Apple", "B-ORG")],
        ]
        model.train(train_data)
        model.save()  # Initial save to rm

    print("\n--- Interactive NER Training ---")
    print("Type a sentence. The model will guess tags.")
    print("If wrong, type the CORRECT tags comma-separated.")
    print("Type 'exit' to quit.\n")

    while True:
        print("Tags: O,B-PER,I-PER,B-LOC,I-LOC,B-ORG,I-ORG")
        print("type \"exit\" to quit.")
        try:
            user_input = input("Enter Text: ")
            if user_input.lower() == 'exit': break
            if not user_input.strip(): continue

            tokens, predicted_tags = model.predict(user_input)

            print(f"\nModel Prediction:")
            formatted_output = " ".join([f"{w}[{t}]" for w, t in zip(tokens, predicted_tags)])
            print(formatted_output)
            print(f"Raw Tags: {','.join(predicted_tags)}")

            correction = input("\nCorrect? (Press Enter if yes, or type tags): ")

            if correction.strip():
                new_tags = [t.strip() for t in correction.split(",")]

                if len(new_tags) != len(tokens):
                    print(f"Error: {len(new_tags)} tags for {len(tokens)} words. Skipped.")
                    continue

                corrected_sentence = list(zip(tokens, new_tags))
                model.train([corrected_sentence], weight=5)
                model.save("ner_model.pkl")  # <--- SAVES AUTOMATICALLY HERE
            else:
                print("--> No changes made.")

        except Exception as e:
            print(f"Something went wrong: {e}")


def inspect_model(model, top_n=5):
    print(f"\n{'=' * 20} MODEL INSPECTION {'=' * 20}")

    print(f"\n[TRANSITIONS] How tags flow into each other:")
    print(f"{'Prev Tag':<10} -> {'Next Tag':<10} : {'Prob':<8} {'(Count)'}")
    print("-" * 45)

    all_trans = []
    for prev, targets in model.transitions.items():
        total_prev = model.word_counts[prev]
        for curr, count in targets.items():
            prob = get_prob(count, total_prev)
            all_trans.append((prob, prev, curr, count))

    for prob, prev, curr, count in sorted(all_trans, reverse=True)[:10]:
        print(f"{prev:<10} -> {curr:<10} : {prob:.4f}   ({count})")

    print(f"\n\n[EMISSIONS] Top words for each tag:")

    tags_to_check = [t for t in model.word_counts.keys() if t != "START"]

    for tag in tags_to_check:
        print(f"\nTAG: {tag}")
        print(f"{'Word':<15} : {'Prob':<8} {'(Count)'}")
        print("-" * 35)

        words = [(w, c) for w, c in model.emissions[tag].items() if not w.startswith("SHAPE:")]

        ranked = []
        total_tag = model.word_counts[tag]
        for w, c in words:
            prob = get_prob(c, total_tag)
            ranked.append((prob, w, c))

        for prob, w, c in sorted(ranked, reverse=True)[:top_n]:
            print(f"{w:<15} : {prob:.4f}   ({c})")

    print(f"\n\n[SHAPES] Top patterns for each tag:")
    for tag in tags_to_check:
        shapes = [(w, c) for w, c in model.emissions[tag].items() if w.startswith("SHAPE:")]

        if not shapes: continue

        print(f"\nTAG: {tag}")
        ranked_shapes = []
        total_shape = model.shape_counts[tag]

        for s, c in shapes:
            clean_shape = s.replace("SHAPE:", "")
            prob = get_prob(c, total_shape)
            ranked_shapes.append((prob, clean_shape, c))

        for prob, s, c in sorted(ranked_shapes, reverse=True)[:top_n]:
            print(f"{s:<15} : {prob:.4f}   ({c})")

    print(f"\n{'=' * 58}\n")