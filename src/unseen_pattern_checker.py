import json


def load_sax_dictionary(dictionary_path):

    with open(dictionary_path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_seen_pattern(sax_word, sax_dictionary):
    """
    Pattern train sözlüğünde varsa seen,
    yoksa unseen kabul edilir.
    """

    return sax_word in sax_dictionary


def check_unseen_patterns(sax_words, sax_dictionary):

    results = []

    for word in sax_words:

        results.append({
            "pattern": word,
            "status": "seen" if is_seen_pattern(word, sax_dictionary) else "unseen"
        })

    return results