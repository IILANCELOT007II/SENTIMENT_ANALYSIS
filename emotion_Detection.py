import torch
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np

df = pd.read_csv('Data/smile-annotations-final.csv', names = ['id', 'text','category'])
df.set_index('id', inplace=True)
#sets the id as index of each observation

df = df[~df.category.str.contains('\|')]
df = df[df.category != 'nocode']

possible_labels = df.category.unique()
label_dict = {}

for index, possible_labels in enumerate(possible_labels):
	label_dict[possible_labels] = index


df['label'] = df.category.replace(label_dict)
#we have replaced the labels with integral values assigned to them in label_dict

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
	df.index.values,
	df.label.values,
	test_size=0.15,
	stratify = df.label.values #makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter
)


df['data_type'] = ['not_set']*df.shape[0]


df.loc[X_train, 'data_type']='train'
df.loc[X_val, 'data_type'] = 'val'
df.groupby(['category', 'label', 'data_type']).count()
#This help in segregating the data inside train and val based on labels


from transformers import BertTokenizer
from torch.utils.data import TensorDataset
#Tokenizer takes raw test and stores it as a token
#token = some random number to identified with

tokenizer = BertTokenizer.from_pretrained(
	'bert-base-uncased',
	do_lower_case = True)

#special tokens represent where the sentence ends
#Attention_mask helps in identifying the empty spaces in the sentence
#since sentences can vary in sizes therefore we pad them them to make the dimensionality constant across all elements
encoded_data_train = tokenizer.batch_encode_plus(
	df[df.data_type == 'train'].text.values,
	add_special_tokens = True,
	return_attention_mask = True,
	pad_to_max_length = True,
	max_length = 256,
	return_tensors = 'pt'
	)

encoded_data_val = tokenizer.batch_encode_plus(
	df[df.data_type == 'val'].text.values,
	add_special_tokens = True,
	return_attention_mask = True,
	pad_to_max_length = True,
	max_length = 256,
	return_tensors = 'pt'
	)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type == 'train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type == 'val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
	'bert-base-uncased',
	num_labels = len(label_dict),
	output_attentions = False,
	output_hidden_states = False)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size= 32

dataloader_train = DataLoader(
	dataset_train, 
	sampler = RandomSampler(dataset_train),
	batch_size = batch_size)

dataloader_val = DataLoader(
	dataset_val, 
	sampler = RandomSampler(dataset_val),
	batch_size = batch_size)

from transformers import AdamW, get_linear_schedule_with_warmup 

optimizer = AdamW(
	model.parameters(),
	lr = 1e-5, #2e-5>5e-5
	eps = 1e-8)

epochs = 10

schedules = get_linear_schedule_with_warmup(
	optimizer,
	num_warmup_steps=0,
	num_training_steps = len(dataloader_train)*epochs)


from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
	preds_flat = np.argmax(pred, axis=1).flatten()
	labels_flat = labels.flatten()
	return f1_score(labels_slat, preds_flat, average='weighted')

def accuracy_per_class(preds, label):
	label_dict_inverse = {v: k for k, v in label_dict.items()}
	preds_flat = np.argmax(preds, axis=1).flatten()

	for label in np.unique(labels_flat):
		y_preds = preds_flat[labels_flat==label]
		y_true = labels_flat[labels_flat==label]
		print(f'Class: {label_dict_inverse[label]}')
		print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)


def evaluate(dataloader_val):
	model.eval()

	loss_val_total = 0
	predictions, true_vals = [], []

	for batch in dataloader_val:
		batch = tuple(b.to(device) for b in batch)
		inputs = {'input_ids':	batch[0],
					'attention_mask':	batch[1],
					'labels':	batch[2],}

		with torch.no_grad():
			outputs = model(**inputs)

		loss = outputs[0]
		logits = outputs[1]
		loss_val_total += loss.item()

		logits = logits.detach().cpu().numpy()
		label_ids = inputs['labels'].cpu().numpy()
		predictions.append(logits)
		true_vals.append(label_ids)

	loss_val_avg = loss_val_total/len(dataloader_val)

	predictions = np.concatenate(predictions, axis=0)
	true_vals = np.concatenate(true_vals, axis=0)

	return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs+1)):

	model.train()
	loss_train_total = 0
	progress_bar = tqdm(dataloader_train,
						desc='Epoch {:1d}'.format(epoch),
						leave = False,
						disable = False)
	for batch in progress_bar:
		model.zero_grad()
		batch = tuple(b.to(device) for b in batch)

		inputs = {
			'input_ids'	: batch[0],
			'attention_mask' : batch[1],
			'labels': batch[2]
		}

		outputs = model(**inputs)

		loss = outputs[0]
		loss_train_total += loss.item()
		loss.backward()
		torch_nn.utils.clips_grad_noem_(model.parameters(), 1.0)

		optimizer.step()
		scheduler.step()

		progress_bar.set_postfix({'training_loss':  '{:.3f}'.format(loss.item()/len(batch))})

	torch.save(model.state_dict(), f'Models/BERT_ft_epoch{epoch}.model')
	torch.write('\nEpoch {epoch}')
	loss_train_avg = loss_train_total/len(dataloader)
	tqdm.write(f'Training loss: {loss_train_avg}')

	val_loss , predictions, true_vals = evaluate(dataloader_val)
	val_f1 = f1_score_func(predictions, true_vals)
	tqdm.write(f'Validation loss: {val_loss}')
	tqdm.write(f'F1 score (weighted): {val_f1}')


model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
	num_labels = len(label_dict),
	output_attentions=False,
	output_hidden_states=False)
model.to(device)
pass

model.load_state_dict(
	torch.load('Models/finetuned_bert_epoch_1_gpu_trained.model',
		map_location = torch.device('cpu')))

_, predictions, true_vals = evaluate(dataloader_val)
accuracy_per_class(predictions, true_vals)