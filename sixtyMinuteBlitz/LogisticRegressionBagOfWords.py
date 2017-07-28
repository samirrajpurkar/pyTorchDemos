from __future__ import print_function
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = [
    ("me gusta comer en la cafeteria".split(), "SPANISH"),
    ("Give it to me".split(), "ENGLISH"),
    ("No creo que sea una buena idea".split(), "SPANISH"),
    ("No it is not a good idea to get lost at sea".split(), "ENGLISH")
]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

word_to_ix = {}
label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

for sentence, _ in data + test_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

# print('len(word_to_ix)', len(word_to_ix))

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):  # calls the init function of nn.Module
        super(BoWClassifier, self).__init__()
        # In this case, we need A and b parameters
        # the parameters of the affine mapping.
        # Torch defines nn.Linear()
        # which provides the affine map.
        # Input dimension is vocab_size
        # Output is num_labels
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax
        return F.log_softmax(self.linear(bow_vec))


def make_bow_vector(p_sentence, p_word_to_ix):
    vec = torch.zeros(len(p_word_to_ix))
    for w in p_sentence:
        vec[p_word_to_ix[w]] += 1
        # print (vec[p_word_to_ix[w]])
    # print (vec.view(1, -1))
    return vec.view(1, -1)


def make_target(p_label, p_label_to_ix):
    # print(torch.LongTensor([p_label_to_ix[p_label]]))
    return torch.LongTensor([p_label_to_ix[p_label]])


def main():
    model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
    # the model knows it parameters.
    # the first output is A, the second is b

    # sample = data[0]
    # # print(sample)
    # bow_vector = make_bow_vector(sample[0], word_to_ix)
    # log_probs = model(autograd.Variable(bow_vector))
    # # print(log_probs)

    # run on test data before we train,
    # just to see a before-and-after

    for instance, label in test_data:
        bow_vector = autograd.Variable(make_bow_vector(instance, word_to_ix))
        log_probs = model(bow_vector)
        # print ("Log P test_data before train", log_probs, label)

    # Print the matrix column corresponding to "creo"
    print("Pre - creo", next(model.parameters())[:, word_to_ix["creo"]])

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        for instance, label in data:
            # Step 1: remmber that pytorch accumulates gradients
            # we need to clear them out before each instance
            model.zero_grad()
            # Step 2:
            # Make our BOW vector and also we must wrap the target in a
            # Variable as an integer.
            # if the target is SPANISH, then we wrap the integer 0.
            # The loss function then knows that the 0th element
            # of log probabilities is the log probability corresponding
            # to SPANISH
            bow_vector = autograd.Variable(make_bow_vector(instance, word_to_ix))
            target = autograd.Variable(make_target(label, label_to_ix))

            # Step 3:
            # run our forward pass
            log_probs = model(bow_vector)

            # Step 4:
            # Compute the loss, gradients, and update the paramaters
            # by calling optimizer.step()
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()

    for instance, label in test_data:
        bow_vector = autograd.Variable(make_bow_vector(instance, word_to_ix))
        log_probs = model(bow_vector)
        print("Log P test_data after train", log_probs, label)

    # Index corresponding to Spanish goes up,
    # English goes down
    print("Post - creo", next(model.parameters())[:, word_to_ix["creo"]])


if __name__ == '__main__':
    main()
