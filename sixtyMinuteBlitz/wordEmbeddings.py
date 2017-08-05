import torch
import torch.autograd as autograd
import  torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        #print('inputs', inputs)
        #print('embeddings', self.embeddings)
        embeds = self.embeddings(inputs).view((1, -1))
        #print('embeds', self.embeddings(inputs))
        #print('embeds', self.embeddings(inputs).view(1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax((out))
        return log_probs

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()

# build a list of tuples.
# Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [(
    [test_sentence[i], test_sentence[i + 1]],
    test_sentence[i + 2]
) for i in range(len(test_sentence) - 2)]
print("First 3 trigrams", trigrams[:3])
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


def main():


    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(100):
        total_loss = torch.Tensor([0])
        for context, target in trigrams[:1]:
            # Step 1.
            # Prepare the inputs to be passed to
            # the model
            # i.e. turn the words into integer indices
            # and wrap them in variables.
            context_idxs = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_idxs))
            print context, target, context_idxs, context_var


            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_var)

            # Step 4:
            # Compute loss function
            # Torch wants the target
            # word wrapped in a variable)
            loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))

            # Step 5:
            # Do the backward pass and update
            # the gradient
            loss.backward()
            optimizer.step()

            total_loss += loss.data

        losses.append((total_loss))

    print(losses)





def learn():
    torch.manual_seed(1)

    word_to_ix = {"hello": 0, "world": 1}
    embeds = nn.Embedding(2, 3)  # 2 words in vocab, 5 dimensional embeddings
    print('embeds', embeds)
    lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
    print('lookup_tensor', lookup_tensor)
    hello_embed = embeds(autograd.Variable(lookup_tensor))
    print('hello_embed', hello_embed)

if __name__ == '__main__':
    main()

