# T38: Compulsory Task 1

import spacy
nlp = spacy.load("en_core_web_md")


# Code extract 1
print("------------Similarity cat, monkey and banana---------------")
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
print("-------------------------------------------------")

"""
NOTE about Code extract 1:
It is interesting to see that the similarity between cat and monkey is higher
than the similarity between monkey and banana, I believe this is because they
are both words that name an animal.
It is also interesting to see that there is a bigger similarity between monkey
and banana than between cat and banana, which means that the fact that we would
naturally associate bananas with monkeys more than we would associate cats with
bananas is being shown here.
"""

# Creating own example of Code extract 1
word1 = nlp("coding")
word2 = nlp("future")
word3 = nlp("job")
word4 = nlp("career")
print("------------Similarity coding, future, job and career---------------")
print(f"{word1} {word2} => {word1.similarity(word2)}")
print(f"{word1} {word3} => {word1.similarity(word3)}")
print(f"{word1} {word4} => {word1.similarity(word4)}")
print(f"{word2} {word3} => {word2.similarity(word3)}")
print(f"{word2} {word4} => {word2.similarity(word4)}")
print(f"{word3} {word4} => {word3.similarity(word4)}")
print("-------------------------------------------------")


# Code extract 2
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# Code extract 3
sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my cat on my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


"""
NOTE 2 on running the example file in the two different language models sm and 
md:
When using the "en_core_web_sm" it seems like the similarity results we
are getting are much lower as opposed to when using the "en_core_web_md".
We are also getting a warning message informing us that the model used is not
loading vectors, which may produce similarity judgments that are not useful.
"""