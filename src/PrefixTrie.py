from pygtrie import CharTrie
from collections import Counter

class PrefixTrie:
    """
    Tries to match testing data to other test data using prefixes
    """


    def run_pred(self, data):
        """
        data is a list of N strings.
        Returns a list of N values, either a string of 3 guesses, or None
        """
        unknown_indices = []
        unknown_data = []
        preds = []
        trie = CharTrie.fromkeys(data, True)
        for inp in data:
            next_chars = Counter(x[len(inp)] for x,_ in trie.iteritems(inp) if len(x) > len(inp))

            # very likely to be a space after punctuation
            if inp[-1] in '.,!?"-':
                next_chars[" "] = 100
            elif not next_chars:
                # no prefixes found, have to run models
                unknown_indices.append(len(preds))
                unknown_data.append(inp)
                preds.append(None)
                continue

            top_guesses = set(c for c,_ in next_chars.most_common(3))

            # add some common characters
            if len(top_guesses) < 3:
                top_guesses.add(' ')
            if len(top_guesses) < 3:
                top_guesses.add('e')
            if len(top_guesses) < 3:
                top_guesses.add('a')
            preds.append(''.join(top_guesses))

        return preds, unknown_indices, unknown_data
