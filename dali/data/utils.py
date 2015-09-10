PUNCTUATION_CHARS = set(list('.,?!-"\'()[]{}:;'))

def split_punctuation(sentence, punctuation=PUNCTUATION_CHARS):
    res = []
    for i, char in enumerate(list(sentence)):
        if char in punctuation:
            if i - 1 >= 0 and sentence[i-1] != ' ':
                res.append(' ')
            res.append(char)
            if i + 1 < len(sentence) and sentence[i + 1] != ' ':
                res.append(' ')
        else:
            res.append(char)
    return ''.join(res)
