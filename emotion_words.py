words = []

for line in open('data/guba-keywords.txt'):
    words.append(line.strip())

words.sort()
with open('data/emo-words.txt', 'w') as f:
    for w in words:
        f.write(w + '\n')