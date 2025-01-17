from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
sentences = ['Stock market is hard to understand.', "No one can predict the stock market."]
inputs = tokenizer(sentences, return_tensors='pt',padding=True,truncation=True) # return_tensors='pt' means returning PyTorch tensors
outputs = bert_model(**inputs)
embeddings = outputs.last_hidden_state

def logical_reasoning(embedding1,embedding2):
    '''
    Takes 2 embeddings and returns whether it is an entailment or neutral.
    Inputs:
        embedding1 -- first embedding of stock market
        embedding2 -- second embedding of stock market
    Returns:
        str: A string of 'entailment' or 'neutral'
    '''
    similarity = (embedding1 * embedding2).sum(dim=-1)
    threshold = 0.8
    if torch.any(similarity > threshold):
        return 'entailment'
    else:
        return 'neutral'


result = logical_reasoning(embeddings[0], embeddings[1])
print(result)

