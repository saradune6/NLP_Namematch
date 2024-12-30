

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime
import random
import numpy as np
import pandas as pd


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")





from datasets import load_dataset
dataset = load_dataset("csv", data_files="traindata.csv")



print(dataset)




from datasets import DatasetDict




print(dataset)

train_testvalid = dataset['train'].train_test_split(test_size=0.2)

test_valid = train_testvalid['test'].train_test_split(test_size=0.5)  # 50% of the 20% becomes 10% of the total


ds = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']
})




ds




dataset=ds




dataset



from transformers import AutoTokenizer, AutoModel


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



class STSBDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):

        # Normalize the similarity scores in the dataset
        similarity_scores = [i['labels'] for i in dataset]
        self.normalized_similarity_scores = [i/5.0 for i in similarity_scores]
        self.first_sentences = [i['name1'] for i in dataset]
        self.second_sentences = [i['name2'] for i in dataset]
        self.concatenated_sentences = [[str(x), str(y)] for x,y in zip(self.first_sentences, self.second_sentences)]

    def __len__(self):
        return len(self.concatenated_sentences)

    def get_batch_labels(self, idx):
        return torch.tensor(self.normalized_similarity_scores[idx])

    def get_batch_texts(self, idx):
        return tokenizer(self.concatenated_sentences[idx], padding='max_length', max_length=128, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


def collate_fn(texts):
    input_ids = texts['input_ids']
    attention_masks = texts['attention_mask']
    features = [{'input_ids': input_id, 'attention_mask': attention_mask}
                for input_id, attention_mask in zip(input_ids, attention_masks)]
    return features


class BertForSTS(torch.nn.Module):

    def __init__(self):
        super(BertForSTS, self).__init__()
        self.bert = models.Transformer('bert-base-uncased', max_seq_length=128)
        self.pooling_layer = models.Pooling(self.bert.get_word_embedding_dimension())
        self.sts_bert = SentenceTransformer(modules=[self.bert, self.pooling_layer])

    def forward(self, input_data):
        output = self.sts_bert(input_data)['sentence_embedding']
        return output


model = BertForSTS()
model.to(device)


class CosineSimilarityLoss(torch.nn.Module):

    def __init__(self,  loss_fn=torch.nn.MSELoss(), transform_fn=torch.nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fn = loss_fn
        self.transform_fn = transform_fn
        self.cos_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, inputs, labels):
        emb_1 = torch.stack([inp[0] for inp in inputs])
        emb_2 = torch.stack([inp[1] for inp in inputs])
        outputs = self.transform_fn(self.cos_similarity(emb_1, emb_2))
        return self.loss_fn(outputs, labels.squeeze())

train_ds = STSBDataset(ds['train'])
val_ds = STSBDataset(ds['valid'])
test_ds = STSBDataset(ds['test'])

train_size = len(train_ds)
val_size = len(val_ds)
test_size = len(test_ds)

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))
print('{:>5,} testing samples'.format(test_size))




batch_size = 10

train_dataloader = DataLoader(
    train_ds,  # The training samples.
    num_workers=4,
    batch_size=batch_size,  # Use this batch size.
    shuffle=True  # Select samples randomly for each batch
)

validation_dataloader = DataLoader(
    val_ds,  # The validation samples.
    num_workers=4,
    batch_size=batch_size  # Use the same batch size
)

test_dataloader = DataLoader(
    test_ds,  # The test samples.
    num_workers=4,
    batch_size=batch_size  # Use the same batch size
)




optimizer = AdamW(model.parameters(),
                  lr = 1e-6)




epochs = 60

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))





def train():
  seed_val = 42

  criterion = CosineSimilarityLoss()
  criterion = criterion.to(device)

  random.seed(seed_val)
  torch.manual_seed(seed_val)

  training_stats = []
  total_t0 = time.time()

  for epoch_i in range(0, epochs):

      # ========================================
      #               Training
      # ========================================

      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      print('Training...')

      t0 = time.time()

      total_train_loss = 0

      model.train()

      for train_data, train_label in tqdm(train_dataloader):

          train_data['input_ids'] = train_data['input_ids'].to(device)
          train_data['attention_mask'] = train_data['attention_mask'].to(device)

          train_data = collate_fn(train_data)
          model.zero_grad()

          output = [model(feature) for feature in train_data]

          loss = criterion(output, train_label.to(device))
          total_train_loss += loss.item()

          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()
          scheduler.step()


      avg_train_loss = total_train_loss / len(train_dataloader)

      # Measure how long this epoch took.
      training_time = format_time(time.time() - t0)

      print("")
      print("  Average training loss: {0:.5f}".format(avg_train_loss))
      print("  Training epoch took: {:}".format(training_time))

      # ========================================
      #               Validation
      # ========================================

      print("")
      print("Running Validation...")

      t0 = time.time()

      model.eval()

      total_eval_accuracy = 0
      total_eval_loss = 0
      nb_eval_steps = 0

      # Evaluate data for one epoch
      for val_data, val_label in tqdm(validation_dataloader):

          val_data['input_ids'] = val_data['input_ids'].to(device)
          val_data['attention_mask'] = val_data['attention_mask'].to(device)

          val_data = collate_fn(val_data)

          with torch.no_grad():
              output = [model(feature) for feature in val_data]

          loss = criterion(output, val_label.to(device))
          total_eval_loss += loss.item()

      # Calculate the average loss over all of the batches.
      avg_val_loss = total_eval_loss / len(validation_dataloader)

      # Measure how long the validation run took.
      validation_time = format_time(time.time() - t0)

      print("  Validation Loss: {0:.5f}".format(avg_val_loss))
      print("  Validation took: {:}".format(validation_time))

      # Record all statistics from this epoch.
      training_stats.append(
          {
              'epoch': epoch_i + 1,
              'Training Loss': avg_train_loss,
              'Valid. Loss': avg_val_loss,
              'Training Time': training_time,
              'Validation Time': validation_time
          }
      )

  print("")
  print("Training complete!")

  print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

  return model, training_stats


model, training_stats = train()


df_stats = pd.DataFrame(data=training_stats)

df_stats = df_stats.set_index('epoch')

df_stats


# In[ ]:


validation_dataloader


# In[44]:


test_dataset = load_dataset("stsb_multi_mt", name="en", split="test")

# Prepare the data
first_sent = [i['sentence1'] for i in test_dataset]
second_sent = [i['sentence2'] for i in test_dataset]
full_text = [[str(x), str(y)] for x,y in zip(first_sent, second_sent)]




dataset



model.eval()

def predict_similarity(sentence_pair):
  test_input = tokenizer(sentence_pair, padding='max_length', max_length = 128, truncation=True, return_tensors="pt").to(device)
  test_input['input_ids'] = test_input['input_ids']
  test_input['attention_mask'] = test_input['attention_mask']
  del test_input['token_type_ids']
  output = model(test_input)
  sim = torch.nn.functional.cosine_similarity(output[0], output[1], dim=0).item()

  return sim



model



train_dataset = pd.DataFrame( dataset['train'] )




test_dataset = pd.DataFrame( dataset['test'] )




val_dataset = pd.DataFrame( dataset['valid'] )



# from datasets import load_dataset
# from sentence_transformers import SentenceTransformer, models
# from transformers import BertTokenizer
# from transformers import get_linear_schedule_with_warmup
# import torch
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import time
# import datetime
# import random
# import numpy as np
# import pandas as pd




# class BertForSTS(torch.nn.Module):

#     def __init__(self):
#         super(BertForSTS, self).__init__()
#         self.bert = models.Transformer('bert-base-uncased', max_seq_length=128)
#         self.pooling_layer = models.Pooling(self.bert.get_word_embedding_dimension())
#         self.sts_bert = SentenceTransformer(modules=[self.bert, self.pooling_layer])

#     def forward(self, input_data):
#         output = self.sts_bert(input_data)['sentence_embedding']
#         return output




# class STSBDataset(torch.utils.data.Dataset):

#     def __init__(self, dataset):

#         # Normalize the similarity scores in the dataset
#         similarity_scores = [i['labels'] for i in dataset]
#         self.normalized_similarity_scores = [i/5.0 for i in similarity_scores]
#         self.first_sentences = [i['name1'] for i in dataset]
#         self.second_sentences = [i['name2'] for i in dataset]
#         self.concatenated_sentences = [[str(x), str(y)] for x,y in zip(self.first_sentences, self.second_sentences)]

#     def __len__(self):
#         return len(self.concatenated_sentences)

#     def get_batch_labels(self, idx):
#         return torch.tensor(self.normalized_similarity_scores[idx])

#     def get_batch_texts(self, idx):
#         return tokenizer(self.concatenated_sentences[idx], padding='max_length', max_length=128, truncation=True, return_tensors="pt")

#     def __getitem__(self, idx):
#         batch_texts = self.get_batch_texts(idx)
#         batch_y = self.get_batch_labels(idx)
#         return batch_texts, batch_y


# def collate_fn(texts):
#     input_ids = texts['input_ids']
#     attention_masks = texts['attention_mask']
#     features = [{'input_ids': input_id, 'attention_mask': attention_mask}
#                 for input_id, attention_mask in zip(input_ids, attention_masks)]
#     return features



# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(f'There are {torch.cuda.device_count()} GPU(s) available.')
#     print('We will use the GPU:', torch.cuda.get_device_name(0))
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")




# def predict_similarity_embedding_model(sentence_pair):
#     """
#     Predict similarity between a pair of sentences
#     """
#     test_input = tokenizer(sentence_pair, padding='max_length', max_length=128, truncation=True, return_tensors="pt").to(device)
#     test_input['input_ids'] = test_input['input_ids']
#     test_input['attention_mask'] = test_input['attention_mask']
#     del test_input['token_type_ids']
#     output = model(test_input)
#     sim = torch.nn.functional.cosine_similarity(output[0], output[1], dim=0).item()
#     return sim



# import pickle
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)
    
# with open('tokenizer.pkl', 'rb') as file:
#     tokenizer = pickle.load(file)




# model.eval()




import pickle

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)



import pickle

with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)







# ### Last but not least, save your model!



PATH = 'bert-sts.pt'
torch.save(model.state_dict(), PATH)


model = BertForSTS()
model.load_state_dict(torch.load(PATH))
model.eval()



