# -*-coding:utf-8 -*-
# @Author  : SATAT

import sys
sys.path.append("../../") #相对路径或绝对路径

import argparse
import os
import numpy as np
from pathlib import Path
from scipy.special import softmax
np.random.seed(1234)
import pickle
import dataloader
from train_classifier import Model
from itertools import zip_longest
import criteria
import random
random.seed(0)
import csv
import time
import string
import math
import json

import joblib
import sys
csv.field_size_limit(sys.maxsize)

import tensorflow_hub as hub

import tensorflow.compat.v1 as tf
import copy
import tensorflow as my_tf2
from tqdm import trange
from InferSent.models import NLINet, InferSent
from esim.model import ESIM
from esim.data import Preprocessor
from esim.utils import correct_predictions
tf.disable_v2_behavior()
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from sklearn.cluster import KMeans
from BERT.tokenization import BertTokenizer
# from BERT.modeling import BertForSequenceClassification, BertConfig, BertForSequenceClassification_embed
from BERT.modeling import BertForSequenceClassification, BertConfig




# tf.compat.v1.disable_eager_execution()
tf.disable_eager_execution()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
version_name = "mvgtext_v2_1"

class NLI_infer_InferSent(nn.Module):
    def __init__(self,
                 pretrained_file,
                 embedding_path,
                 data,
                 batch_size=32):
        super(NLI_infer_InferSent, self).__init__()

        config_nli_model = {
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'n_enc_layers': 1,
            'dpout_model': 0.,
            'dpout_fc': 0.,
            'fc_dim': 512,
            'bsize': batch_size,
            'n_classes': 3,
            'pool_type': 'max',
            'nonlinear_fc': 0,
            'encoder_type': 'InferSent',
            'use_cuda': True,
            'use_target': False,
            'version': 1,
        }
        params_model = {'bsize': 64, 'word_emb_dim': 200, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}

        print("\t* Building model...")
        self.model = NLINet(config_nli_model).cuda()
        print("Reloading pretrained parameters...")
        self.model.load_state_dict(torch.load(os.path.join("savedir/", "model.pickle")))
        print('Building vocab and embeddings...')
        self.dataset = NLIDataset_InferSent(embedding_path, data=data, batch_size=batch_size)
    def text_pred(self, text_data):
        self.model.eval()
        data_batches = self.dataset.transform_text(text_data)
        probs_all = []
        with torch.no_grad():
            for batch in data_batches:
                (s1_batch, s1_len), (s2_batch, s2_len) = batch
                s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

class NLI_infer_ESIM(nn.Module):
    def __init__(self,
                 pretrained_file,
                 worddict_path,
                 local_rank=-1,
                 batch_size=32):
        super(NLI_infer_ESIM, self).__init__()

        self.batch_size = batch_size
        self.device = torch.device("cuda:{}".format(local_rank) if local_rank > -1 else "cuda")
        checkpoint = torch.load(pretrained_file)
        vocab_size = checkpoint['model']['_word_embedding.weight'].size(0)
        embedding_dim = checkpoint['model']['_word_embedding.weight'].size(1)
        hidden_size = checkpoint['model']['_projection.0.weight'].size(0)
        num_classes = checkpoint['model']['_classification.4.weight'].size(0)

        print("\t* Building model...")
        self.model = ESIM(vocab_size,
                          embedding_dim,
                          hidden_size,
                          num_classes=num_classes,
                          device=self.device).to(self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.dataset = NLIDataset_ESIM(worddict_path)

    def text_pred(self, text_data):
        self.model.eval()
        device = self.device

        self.dataset.transform_text(text_data)
        dataloader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size)

        probs_all = []
        with torch.no_grad():
            for batch in dataloader:
                premises = batch['premise'].to(device)
                premises_lengths = batch['premise_length'].to(device)
                hypotheses = batch['hypothesis'].to(device)
                hypotheses_lengths = batch['hypothesis_length'].to(device)

                _, probs = self.model(premises,
                                      premises_lengths,
                                      hypotheses,
                                      hypotheses_lengths)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=3).cuda()

        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data):
        self.model.eval()

        dataloader = self.dataset.transform_text(text_data)

        probs_all = []
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
       
        module_url = "/mnt/1TB-SSD-DATA/sy/USE/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


def read_data(filepath, data_size, target_model='infersent', lowercase=False, ignore_punctuation=False, stopwords=[]):
    if target_model == 'bert':
        labeldict = {"contradiction": 0,
                     "entailment": 1,
                     "neutral": 2}
    else:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}
    with open(filepath, 'r', encoding='utf8') as input_data:
        premises, hypotheses, labels = [], [], []
        punct_table = str.maketrans({key: ' '
                                     for key in string.punctuation})

        for idx, line in enumerate(input_data):
            if idx >= data_size:
                break

            line = line.strip().split('\t')
            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            premises.append([w for w in premise.rstrip().split()
                             if w not in stopwords])
            hypotheses.append([w for w in hypothesis.rstrip().split()
                               if w not in stopwords])
            if filepath.split("/")[-1] == "mnli":
                labeldict = {'0':1,'1':2,'2':0}
                labels.append(labeldict[line[0]])
            else:
                labels.append(labeldict[line[0]])
            # labels.append(labeldict[line[0]])

        return {"premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


class NLIDataset_ESIM(Dataset):
    def __init__(self,
                 worddict_path,
                 padding_idx=0,
                 bos="_BOS_",
                 eos="_EOS_"):
        self.bos = bos
        self.eos = eos
        self.padding_idx = padding_idx

        with open(worddict_path, 'rb') as pkl:
            self.worddict = pickle.load(pkl)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
            "premise": self.data["premises"][index],
            "premise_length": min(self.premises_lengths[index],
                                  self.max_premise_length),
            "hypothesis": self.data["hypotheses"][index],
            "hypothesis_length": min(self.hypotheses_lengths[index],
                                     self.max_hypothesis_length)
        }

    def words_to_indices(self, sentence):
        indices = []
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                index = self.worddict['_OOV_']
            indices.append(index)
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def transform_to_indices(self, data):
        transformed_data = {"premises": [],
                            "hypotheses": []}

        for i, premise in enumerate(data['premises']):
            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def transform_text(self, data):
        data = self.transform_to_indices(data)

        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {
            "premises": torch.ones((self.num_sequences,
                                    self.max_premise_length),
                                   dtype=torch.long) * self.padding_idx,
            "hypotheses": torch.ones((self.num_sequences,
                                      self.max_hypothesis_length),
                                     dtype=torch.long) * self.padding_idx}

        for i, premise in enumerate(data["premises"]):
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])


class NLIDataset_InferSent(Dataset):

    def __init__(self,
                 embedding_path,
                 data,
                 word_emb_dim=300,
                 batch_size=32,
                 bos="<s>",
                 eos="</s>"):

        self.bos = bos
        self.eos = eos
        self.word_emb_dim = word_emb_dim
        self.batch_size = batch_size

        self.word_vec = self.build_vocab(data['premises'] + data['hypotheses'], embedding_path)

    def build_vocab(self, sentences, embedding_path):
        word_dict = self.get_word_dict(sentences)
        word_vec = self.get_embedding(word_dict, embedding_path)
        print('Vocab size : {0}'.format(len(word_vec)))
        return word_vec

    def get_word_dict(self, sentences):
        word_dict = {}
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<oov>'] = ''
        return word_dict

    def get_embedding(self, word_dict, embedding_path):
        word_vec = {}
        word_vec['<oov>'] = np.random.normal(size=(self.word_emb_dim))
        with open(embedding_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with embedding vectors'.format(
            len(word_vec), len(word_dict)))
        return word_vec

    def get_batch(self, batch, word_vec, emb_dim=300):
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        embed = np.zeros((max_len, len(batch), emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] in word_vec:
                    embed[j, i, :] = word_vec[batch[i][j]]
                else:
                    embed[j, i, :] = word_vec['<oov>']

        return torch.from_numpy(embed).float(), lengths

    def transform_text(self, data):
        premises = data['premises']
        hypotheses = data['hypotheses']

        premises = [['<s>'] + premise + ['</s>'] for premise in premises]
        hypotheses = [['<s>'] + hypothese + ['</s>'] for hypothese in hypotheses]

        batches = []
        for stidx in range(0, len(premises), self.batch_size):
            s1_batch, s1_len = self.get_batch(premises[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            s2_batch, s2_len = self.get_batch(hypotheses[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            batches.append(((s1_batch, s1_len), (s2_batch, s2_len)))

        return batches


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):

        features = []
        for (ex_index, (text_a, text_b)) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            tokens_b = None
            if text_b:
                tokens_b = tokenizer.tokenize(' '.join(text_b))
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data):
        eval_features = self.convert_examples_to_features(list(zip(data['premises'], data['hypotheses'])),
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

        return eval_dataloader


def calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor):
    len_text = len(text_ls)
    half_sim_score_window = (sim_score_window - 1) // 2

    if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = idx - half_sim_score_window
        text_range_max = idx + half_sim_score_window + 1
    elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = 0
        text_range_max = sim_score_window
    elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
        text_range_min = len_text - sim_score_window
        text_range_max = len_text
    else:
        text_range_min = 0
        text_range_max = len_text

    if text_range_min < 0:
        text_range_min = 0
    if text_range_max > len_text:
        text_range_max = len_text

    if idx == -1:
        text_rang_min = 0
        text_range_max = len_text

    semantic_sims = \
        sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
                                   list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

    return semantic_sims


def get_attack_result(hypotheses, premise, predictor, orig_label, batch_size):
    new_probs = predictor({'premises': [premise] * len(hypotheses), 'hypotheses': hypotheses})
    pr = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
    return pr


def combined_similarity(vector, vectors, k=3):
    vector_reshape = vector.reshape(-1)
    vectors_reshape = vectors.reshape(vectors.shape[0],vector_reshape.shape[0])
    cosine_similarities = np.dot(vectors_reshape, vector_reshape) / (np.linalg.norm(vector_reshape) * np.linalg.norm(vectors_reshape, axis=1))
    cosine_similarities = 0.5 * (cosine_similarities + 1)
    # 耗时特短
    combined_scores = cosine_similarities
    sorted_score = np.argsort(combined_scores)
    if len(sorted_score) == 1:
        most_similar_index = sorted_score[-1]
    else:
        # most_similar_index = sorted_score[-1-k:-1]
        most_similar_index = sorted_score[-1-k:-1]
    
    return most_similar_index, combined_scores

# 换回，添加了success_dict 和 fail_list
def back_to_ori_word(random_text, premise, text_ls, qrs, success_attack_dict, fail_attack_list, budget, sim_score_window,
                     sim_predictor, predictor, orig_label, batch_size):
    # STEP 2: Search Space Reduction i.e.  Move Sample Close to Boundary
    while True:
        choices = []
        # For each word substituted in the original text, change it with its original word and compute
        # the change in semantic similarity.
        for i in range(len(text_ls)):
            if random_text[i] != text_ls[i]:
                new_text = random_text[:]
                new_text[i] = text_ls[i]
                semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                if " ".join(new_text) in success_attack_dict.keys():
                    choices.append((i, semantic_sims[0]))
                elif " ".join(new_text) in fail_attack_list:
                    continue
                else:
                    qrs += 1
                    pr = get_attack_result([new_text], premise,  predictor, orig_label, batch_size)
                    if np.sum(pr) > 0:
                        success_attack_dict[" ".join(new_text)] = semantic_sims[0]
                        choices.append((i, semantic_sims[0]))
                    else:
                        fail_attack_list.append(" ".join(new_text))
                    if qrs >= budget:
                        return random_text, qrs, success_attack_dict, fail_attack_list

        # Sort the relacements by semantic similarity and replace back the words with their original
        # counterparts till text remains adversarial.
        if len(choices) > 0:
            choices.sort(key=lambda x: x[1])
            choices.reverse()
            for i in range(len(choices)):
                new_text = random_text[:]
                new_text[choices[i][0]] = text_ls[choices[i][0]]
                if " ".join(new_text) not in success_attack_dict.keys() and " ".join(new_text) not in fail_attack_list:
                    pr = get_attack_result([new_text], premise,  predictor, orig_label, batch_size)
                    qrs += 1
                    if pr[0] == 0:
                        fail_attack_list.append(" ".join(new_text))
                        if qrs >= budget:
                            return random_text, qrs, success_attack_dict, fail_attack_list
                        else:
                            break
                    random_text[choices[i][0]] = text_ls[choices[i][0]]
                    semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                    success_attack_dict[" ".join(new_text)] = semantic_sims[0]
                    if qrs >= budget:
                        return random_text, qrs, success_attack_dict, fail_attack_list
                elif " ".join(new_text) in success_attack_dict.keys():
                    # print("换回到一个成功样本，更新样本！！！！")
                    random_text[choices[i][0]] = text_ls[choices[i][0]]  

        if len(choices) == 0:
            break

    return random_text, qrs, success_attack_dict, fail_attack_list

def is_budget(sim_score_window, sim_predictor,budget,synonyms_dict,batch_size,text_ls,success_attack_dict,fail_attack_list,changed_indices,random_sim,orig_label,predictor,qrs):
   
    update_attack_list_sim_pert_weight = list(
        sorted(success_attack_dict.items(), key=lambda item: item[1], reverse=True))[:100]

    sim_pert_weight_dict = {}
    for one in update_attack_list_sim_pert_weight:
        change_one = 0
        text_one = one[0].split(" ")
        for i in range(len(text_ls)):
            if text_ls[i] != text_one[i]:
                change_one += 1
        sim_pert_weight_dict[one[0]] = change_one - one[1]

    new_sample = list(sorted(sim_pert_weight_dict.items(), key=lambda item: item[1], reverse=False))[0]
    new_text = new_sample[0]
    new_sim = new_sample[1]

    best_attack = new_text.split(" ")
    best_sim = new_sim
    choices = []
    check = dict()
    for i in range(len(text_ls)):
        if text_ls[i] != best_attack[i]:
            now_best = best_attack[:]
            now_best[i] = text_ls[i]
            semantic_sims = calc_sim(text_ls, [now_best], -1, sim_score_window, sim_predictor)
            choices.append((i, semantic_sims[0]))  # 优化顺序
            check[str(i)] = True
    print("修改单词数：", len(choices))
    choices_list = choices[:]

    print("-------------------开始单样本优化！-------------------------")
    while qrs < budget:
        replaced_txt = []
        get_new = False
        for i, _ in choices_list:
            for j in synonyms_dict[text_ls[i]]:
                tmp_txt = best_attack.copy()
                tmp_txt[i] = j
                sims = calc_sim(text_ls, [tmp_txt], -1, sim_score_window, sim_predictor)
                if sims[0] > best_sim:
                    replaced_txt.append([tmp_txt, sims[0]])

        if len(replaced_txt) > 0:
            replaced_txt.sort(key=lambda x: x[1])
            replaced_txt.reverse()
            candi_samples_filter = replaced_txt
            print("倒排sim找最大sim,candi_samples_filter中样本个数：", len(candi_samples_filter), "qrs:", qrs)

            for i, sim in candi_samples_filter:
                if " ".join(i) not in success_attack_dict.keys() and " ".join(i) not in fail_attack_list:
                    pr = get_attack_result([i],premise, predictor, orig_label, batch_size)
                    qrs += 1
                    if np.sum(pr) > 0:
                        success_attack_dict[" ".join(i)] = sim
                        if sim > best_sim:
                            for x in range(len(text_ls)):
                                if i[x] != best_attack[x]:
                                    check[str(x)] = False
                                    get_new = True

                            best_attack = i
                            best_sim = sim
                        break
                    else:
                        fail_attack_list.append(i)
                    if qrs >= budget:
                        break

        all_values_false = all(value == False for value in check.values())

        if all_values_false:
            break
        elif not get_new:
            break

    pr = get_attack_result([best_attack], premise, predictor, orig_label, batch_size)
    best_sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]
    print("-------------------单样本优化结束-------------------")
    print("样本总数：", len(success_attack_dict.keys()) + len(fail_attack_list), "查询总数：", qrs, "best sim：", best_sim)

    del success_attack_dict
    del fail_attack_list

    # return
    max_changes = 0
    for i in range(len(text_ls)):
        if text_ls[i] != best_attack[i]:
            max_changes += 1

    now_label = torch.argmax(predictor({'premises':[premise], 'hypotheses': [best_attack]})),


    random_changed = 0
    if len(changed_indices) == 0:
        random_changed = max_changes
    else:
        random_changed = len(changed_indices)

    return ' '.join(best_attack), max_changes, random_changed,orig_label,now_label, qrs, best_sim, random_sim

def mvgtext_attack(
            fuzz_val, top_k_words, sample_index, hypotheses, premise,
            true_label, predictor, stop_words_set, word2idx, idx2word,
            cos_sim, sim_predictor=None, import_score_threshold=-1.,
            sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
            batch_size=32,embed_func = '',budget=1000,myargs=None):

    orig_probs = predictor({'premises': [premise], 'hypotheses': [hypotheses]}).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        return '', 0, 0, orig_label, orig_label, 0, 0, 0
    else:
        # word2idx 构建
        word_idx_dict={}
        with open(embed_func, 'r') as ifile:
            for index, line in enumerate(ifile):
                word = line.strip().split()[0]
                word_idx_dict[word] = index
        embed_file=open(embed_func)
        embed_content=embed_file.readlines()
        text_ls = hypotheses[:]
        pos_ls = criteria.get_pos(text_ls)
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1
        
        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        for i in range(len(pos_ls)):
            if pos_ls[i] in pos_pref and len(text_ls[i]) > 2:
                words_perturb.append((i, text_ls[i]))

        random.shuffle(words_perturb)
        words_perturb = words_perturb[:top_k_words]

        # get words perturbed idx embed doc_idx.find synonyms and make a dict of synonyms of each word.
        words_perturb_idx= []
        words_perturb_embed = []
        words_perturb_doc_idx = []
        for idx, word in words_perturb:
            if word in word_idx_dict:
                words_perturb_doc_idx.append(idx)
                words_perturb_idx.append(word2idx[word])
                words_perturb_embed.append([float(num) for num in embed_content[ word_idx_dict[word] ].strip().split()[1:]])

        words_perturb_embed_matrix = np.asarray(words_perturb_embed)

        synonym_words,synonym_values=[],[]
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp=[]
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            temp=[]
            for ii in res[0]:
                temp.append(ii)
            synonym_values.append(temp)

        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms

        # 初始化必要参数--------------------------------
        qrs = 0  # 查询次数
        num_changed = 0  # 单词修改数
        th = 0  # thershold
        alpha = 5  # 设置初始化有几个对抗样本

        update_attack_list = []  # (text,sim)     alpha_now个当前最佳的样本（相似度降序，alpha_now≤alpha）
        success_attack_dict = dict()  # {text:sim}    已查询成功
        fail_attack_list = []  # (text)     丢弃

        now_adv_n = 0
        '''
        初始化Step1：局部单词替换
        '''
        while qrs < len(text_ls):
            random_text = text_ls[:]
            for i in range(len(synonyms_all)):
                idx = synonyms_all[i][0]
                syn = synonyms_all[i][1]
                random_text[idx] = random.choice(syn)
                if i >= th:
                    break
            if th > len_text:
                break
            pr = get_attack_result([random_text],premise, predictor, orig_label, batch_size)
            qrs += 1
            th += 1
            if np.sum(pr) > 0:
                semantic_sims = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)
                now_adv_n += 1
                success_attack_dict[" ".join(random_text)] = semantic_sims[0]
                break
            else:
                fail_attack_list.append(" ".join(random_text))

        old_qrs = qrs

        '''
        初始化Step2：全局随机替换
        '''
        while qrs < old_qrs + 2500 and now_adv_n == 0:
            random_text = text_ls[:]
            for j in range(len(synonyms_all)):
                idx = synonyms_all[j][0]
                syn = synonyms_all[j][1]
                random_text[idx] = random.choice(syn)
                if j >= len_text:
                    break
            pr = get_attack_result([random_text], premise, predictor, orig_label, batch_size)
            qrs += 1
            if np.sum(pr) > 0:
                semantic_sims = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)
                now_adv_n += 1
                success_attack_dict[" ".join(random_text)] = semantic_sims[0]
                break
            else:
                fail_attack_list.append(" ".join(random_text))

        '''
        初始化Step3：全局固定单词替换，49个同义词同步初始化alpha个对抗样本-
        '''
        for syn_index in range(1, 50):
            random_text = text_ls[:]
            for j in range(len(synonyms_all)):
                idx = synonyms_all[j][0]
                syn = synonyms_all[j][1]
                random_text[idx] = syn[syn_index]
            pr = get_attack_result([random_text], premise, predictor, orig_label, batch_size)
            qrs += 1
            if np.sum(pr) > 0:
                semantic_sims = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)
                now_adv_n += 1
                success_attack_dict[" ".join(random_text)] = semantic_sims[0]
                if now_adv_n >= alpha:
                    print("初始化成功，已找到alpha个样本，查询次数为：", syn_index)
                    break
            else:
                fail_attack_list.append(" ".join(random_text))

        # 初始化结束
        if now_adv_n < 1:
            return ' '.join(random_text), 0, 0, \
                   orig_label, orig_label, qrs, 0, 0
        else:
            keys = success_attack_dict.keys()
            for one in list(keys):
                new_text, qrs, success_attack_dict, fail_attack_list, = back_to_ori_word(
                    one.split(" "),premise,  text_ls, qrs, success_attack_dict,
                    fail_attack_list, budget, sim_score_window,
                    sim_predictor, predictor, orig_label, batch_size)
                if qrs >= (budget * 0.8):
                    new_sim = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)[0]
                    return is_budget(sim_score_window, sim_predictor, budget, synonyms_dict, batch_size,
                                     text_ls, success_attack_dict, fail_attack_list, [], new_sim, orig_label,
                                     predictor, qrs)

                if ' '.join(new_text) not in fail_attack_list and ' '.join(
                        new_text) not in success_attack_dict.keys():
                    pr = get_attack_result([new_text], premise, predictor, orig_label, batch_size)
                    qrs += 1
                    if np.sum(pr) > 0:
                        semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        success_attack_dict[" ".join(random_text)] = semantic_sims[0]
                    else:
                        fail_attack_list.append(" ".join(random_text))

                    if qrs >= (budget * 0.8):
                        new_sim = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)[0]
                        return is_budget(sim_score_window, sim_predictor, budget, synonyms_dict, batch_size,
                                         text_ls, success_attack_dict, fail_attack_list, [], new_sim,
                                         orig_label, predictor, qrs)

            update_attack_list = list(
                sorted(success_attack_dict.items(), key=lambda item: item[1], reverse=True))[:alpha]

            changed_indices = []  
            num_changed = 0
            best_attack = update_attack_list[0][0].split(" ")
            for i in range(len(text_ls)):
                if text_ls[i] != best_attack[i]:
                    changed_indices.append(i)
                    num_changed += 1

            random_sim = update_attack_list[0][1]
            best_sim = random_sim

            only_one = False
            if num_changed == 1:  
                only_one = True
                
            else:
                search_times = 0
                search_th = 40 
                gamma = 20
                tt = 0
                search_sim_th = -2.0
                while search_times < search_th and tt < 20:

                    tt += 1 
                    best_adv_embed = []
                    for idx in words_perturb_doc_idx:
                        best_adv_embed.append(
                            [float(num) for num in
                             embed_content[word_idx_dict[best_attack[idx]]].strip().split()[1:]])
                    best_adv_embed_matrix = np.asarray(best_adv_embed)
                    best_noise = best_adv_embed_matrix - words_perturb_embed_matrix
                    tmp = best_attack[:]

                    tmp_noise = np.zeros_like(best_noise)
                    for _ in range(gamma):
                        update_text = text_ls[:]
                        old_update_attack_list = list(
                            sorted(success_attack_dict.items(), key=lambda item: item[1], reverse=True))[:alpha]

                        for j, _ in update_attack_list:
                            j_text_list = j.split(" ")
                            now_adv_embed = []  
                            for idx in words_perturb_doc_idx:
                                now_adv_embed.append(
                                    [float(num) for num in
                                     embed_content[word_idx_dict[j_text_list[idx]]].strip().split()[1:]])
                            now_adv_embed_matrix = np.asarray(now_adv_embed)
                            tmp_noise = tmp_noise + now_adv_embed_matrix - words_perturb_embed_matrix
                        current_noise = tmp_noise / len(update_attack_list)

                        u = np.random.uniform(0.5, 1.0)
                        new_noise = u * best_noise + (1 - u) * current_noise
                        new_noise_d = new_noise - np.random.normal(loc=0.0, scale=1, size=new_noise.shape)

                        nonzero_ele = np.nonzero(np.linalg.norm(new_noise, axis=-1))[0].tolist()

                        perturb_word_idx_list = nonzero_ele
                        for perturb_idx in range(len(perturb_word_idx_list)):
                            perturb_word_idx = perturb_word_idx_list[perturb_idx]
                            perturb_target = words_perturb_embed_matrix[perturb_word_idx] + new_noise_d[
                                perturb_word_idx]  # 降噪后
                            syn_feat_set = []
                            for syn in synonyms_all[perturb_word_idx][1]:
                                syn_feat = [float(num) for num in
                                            embed_content[word_idx_dict[syn]].strip().split()[1:]]
                                syn_feat_set.append(syn_feat)

                            perturb_syn_dist = np.sum((syn_feat_set - perturb_target) ** 2, axis=1)
                            perturb_syn_order = np.argsort(perturb_syn_dist)
                            replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]

                            update_text[synonyms_all[perturb_word_idx][0]] = replacement

                        if ' '.join(update_text) not in fail_attack_list and ' '.join(
                                update_text) not in success_attack_dict.keys():
                            now_sim = float(
                                calc_sim(text_ls, [update_text], -1, sim_score_window, sim_predictor)[0])
                            if now_sim > search_sim_th:
                                pr = get_attack_result([update_text], premise, predictor, orig_label, batch_size)
                                qrs += 1

                                if np.sum(pr) > 0:
                                    semantic_sims = calc_sim(text_ls, [update_text], -1, sim_score_window,
                                                             sim_predictor)
                                    success_attack_dict[" ".join(update_text)] = semantic_sims[0]
                                    print("根据动量和方差，找到的样本攻击成功：", update_text, "当前：", semantic_sims[0], "最佳：",
                                          best_sim,
                                          "查询次数：", qrs)
                                    new_text, qrs, success_attack_dict, fail_attack_list, = back_to_ori_word(
                                        update_text,premise,  text_ls, qrs, success_attack_dict, fail_attack_list,
                                        budget, sim_score_window,
                                        sim_predictor, predictor, orig_label, batch_size)

                                    if qrs >= (budget * 0.8):
                                        return is_budget(sim_score_window, sim_predictor, budget, synonyms_dict,
                                                         batch_size, text_ls, success_attack_dict,
                                                         fail_attack_list, changed_indices, random_sim,
                                                         orig_label, predictor, qrs)

                                    if ' '.join(new_text) not in fail_attack_list and ' '.join(
                                            new_text) not in success_attack_dict.keys():
                                        print("!!!Error!!!不可能，绝对不可能!!!") 
                                        pr = get_attack_result([new_text], premise, predictor, orig_label, batch_size)
                                        qrs += 1
                                        if np.sum(pr) > 0:
                                            semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window,
                                                                     sim_predictor)
                                            success_attack_dict[" ".join(new_text)] = semantic_sims[0]
                                        else:
                                            fail_attack_list.append(" ".join(new_text))
                                        if qrs >= (budget * 0.8):
                                            return is_budget(sim_score_window, sim_predictor, budget,
                                                             synonyms_dict, batch_size, text_ls,
                                                             success_attack_dict, fail_attack_list,
                                                             changed_indices, random_sim, orig_label, predictor,
                                                             qrs)
                                else:
                                    fail_attack_list.append(" ".join(update_text))
                                if qrs >= (budget * 0.8):
                                    return is_budget(sim_score_window, sim_predictor, budget, synonyms_dict,
                                                     batch_size, text_ls, success_attack_dict, fail_attack_list,
                                                     changed_indices, random_sim, orig_label, predictor, qrs)
                       
                        new_update_attack_list = list(
                            sorted(success_attack_dict.items(), key=lambda item: item[1], reverse=True))[:alpha]

                        if new_update_attack_list == old_update_attack_list:
                            search_times += 1
                           
                        else:
                            # print("前alpha不一样,当前search_times=", search_times)
                            search_times = 0
                            # print("search_times归零:", search_times)

                        best_attack = new_update_attack_list[0][0].split(" ")
                        best_sim = new_update_attack_list[0][1]

                        update_attack_list_sim_pert_weight = list(
                            sorted(success_attack_dict.items(), key=lambda item: item[1], reverse=True))[:100]
                        search_sim_th = update_attack_list_sim_pert_weight[-1][1]
                        # print("search_sim_th:", search_sim_th)

                        sim_pert_weight_dict = {}
                        for one in update_attack_list_sim_pert_weight:
                            change_one = 0
                            text_one = one[0].split(" ")
                            for i in range(len(text_ls)):
                                if text_ls[i] != text_one[i]:
                                    change_one += 1
                            sim_pert_weight_dict[one[0]] = change_one - one[1]

                        update_attack_list = list(
                            sorted(sim_pert_weight_dict.items(), key=lambda item: item[1], reverse=False))[
                                             :alpha]

                    print("第", tt, "次迭代，内循环gamma结束，更新best attack：", best_attack, "更新best sim：", best_sim,
                          "查询次数：", qrs)
                print("多样本搜索结束，外循环迭代次数：", tt)
            print('成功样本数：', len(success_attack_dict.keys()), '失败样本数：', len(fail_attack_list))
            print("样本总数：", len(success_attack_dict.keys()) + len(fail_attack_list), "查询总数：", qrs, "best sim：",
                  best_sim)

            if qrs >= (budget * 0.8):
                return is_budget(sim_score_window, sim_predictor, budget, synonyms_dict, batch_size, text_ls,
                                 success_attack_dict, fail_attack_list, changed_indices, random_sim,
                                 orig_label, predictor, qrs)

            if not only_one:
                success_best = \
                list(sorted(sim_pert_weight_dict.items(), key=lambda item: item[1], reverse=False))[0]
                success_best_text = success_best[0]

            else:
                success_best = \
                list(sorted(success_attack_dict.items(), key=lambda item: item[1], reverse=True))[0]
                success_best_text = success_best[0]

            best_attack = success_best_text.split(" ")
            best_sim = success_attack_dict[success_best_text]
            choices = []
            check = dict()
            for i in range(len(text_ls)):
                if text_ls[i] != best_attack[i]:
                    now_best = best_attack[:]
                    now_best[i] = text_ls[i]
                    semantic_sims = calc_sim(text_ls, [now_best], -1, sim_score_window, sim_predictor)
                    choices.append((i, semantic_sims[0]))
                    check[str(i)] = True
            print("修改单词数：", len(choices))
            choices_list = choices[:]

            print("-------------------开始单样本优化！-------------------------")

            while True:
                replaced_txt = []
                get_new = False
                for i, _ in choices_list:

                    for j in synonyms_dict[text_ls[i]]:
                        # print(j)
                        tmp_txt = best_attack.copy()
                        tmp_txt[i] = j
                        sims = calc_sim(text_ls, [tmp_txt], -1, sim_score_window, sim_predictor)
                        if sims[0] > best_sim:
                            replaced_txt.append([tmp_txt, sims[0]])

                if len(replaced_txt) > 0:
                    replaced_txt.sort(key=lambda x: x[1])
                    replaced_txt.reverse()
                    candi_samples_filter = replaced_txt
                    print("倒排sim找最大sim,candi_samples_filter中样本个数：", len(candi_samples_filter), "qrs:", qrs)

                    for i, sim in candi_samples_filter:
                        if " ".join(i) not in success_attack_dict.keys() and " ".join(
                                i) not in fail_attack_list:
                            pr = get_attack_result([i], premise, predictor, orig_label, batch_size)
                            qrs += 1
                            if np.sum(pr) > 0:
                                success_attack_dict[" ".join(i)] = sim
                                if sim > best_sim:
                                    for x in range(len(text_ls)):
                                        if i[x] != best_attack[x]:
                                            check[str(x)] = False
                                            get_new = True

                                    best_attack = i
                                    best_sim = sim
                                    print("单样本优化更新啦！！！！当前sim:", best_sim)
                                break
                            else:
                                fail_attack_list.append(i)
                            if qrs >= (budget * 0.8):
                                return is_budget(sim_score_window, sim_predictor, budget, synonyms_dict,
                                                 batch_size, text_ls, success_attack_dict, fail_attack_list,
                                                 changed_indices,
                                                 random_sim, orig_label, predictor, qrs)

                all_values_false = all(value == False for value in check.values())

                if all_values_false:
                    break
                elif not get_new:
                    break

            pr = get_attack_result([best_attack], premise, predictor, orig_label, batch_size)

            print("-------------------单样本优化结束-------------------")
            print("样本总数：", len(success_attack_dict.keys()) + len(fail_attack_list), "查询总数：", qrs, "best sim：",
                  best_sim)

     

            del success_attack_dict
            del fail_attack_list

            # return
            max_changes = 0
            for i in range(len(text_ls)):
                if text_ls[i] != best_attack[i]:
                    max_changes += 1

            return ' '.join(best_attack), max_changes, len(changed_indices), \
                   orig_label, torch.argmax(predictor({'premises':[premise], 'hypotheses': [best_attack]})),qrs, best_sim, random_sim


def main():
    # if True 方便看代码
    if True:
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset_path",
                            type=str,
                            # required=True,
                            default="/mnt/1TB-SSD-DATA/sy/hard_label/data/mr",
                            help="Which dataset to attack.")
        parser.add_argument("--target_model",
                            type=str,
                            required=True,
                            choices=['infersent', 'esim', 'bert'],
                            help="Target models for text classification: fasttext, charcnn, word level lstm "
                                 "For NLI: InferSent, ESIM, bert-base-uncased")


        parser.add_argument("--target_model_path",
                            type=str,
                            # required=True,
                            default="/mnt/1TB-SSD-DATA/sy/hard_label/model/cnn/mr",
                            help="pre-trained target model path")
        parser.add_argument("--word_embeddings_path",
                            type=str,
                            default='/mnt/1TB-SSD-DATA/sy/hard_label/other/glove.6B.200d.txt',
                            help="path to the word embeddings for the target model")
        parser.add_argument("--counter_fitting_embeddings_path",
                            type=str,
                            default="/mnt/1TB-SSD-DATA/sy/hard_label/other/counter-fitted-vectors.txt",
                            help="path to the counter-fitting embeddings we used to find synonyms")
        parser.add_argument("--counter_fitting_cos_sim_path",
                            type=str,
                            default='/mnt/1TB-SSD-DATA/sy/hard_label/other/mat.txt',
                            help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
        parser.add_argument("--USE_cache_path",
                            type=str,
                            # required=True,
                            default="",
                            help="Path to the USE encoder cache.")
        parser.add_argument("--output_dir",
                            type=str,
                            default='nli_adv_results',
                            help="The output directory where the attack results will be written.")

        parser.add_argument("--sim_score_window",
                            default=40,
                            type=int,
                            help="Text length or token number to compute the semantic similarity score")
        parser.add_argument("--import_score_threshold",
                            default=-1.,
                            type=float,
                            help="Required mininum importance score.")
        parser.add_argument("--sim_score_threshold",
                            default=0.7,
                            type=float,
                            help="Required minimum semantic similarity score.")
        parser.add_argument("--synonym_num",
                            default=50,
                            type=int,
                            help="Number of synonyms to extract")
        parser.add_argument("--batch_size",
                            default=32,
                            type=int,
                            help="Batch size to get prediction")
        parser.add_argument("--data_size",
                            default=1000,
                            type=int,
                            help="Data size to create adversaries")
        parser.add_argument("--perturb_ratio",
                            default=0.,
                            type=float,
                            help="Whether use random perturbation for ablation study")
        parser.add_argument("--max_seq_length",
                            default=128,
                            type=int,
                            help="max sequence length for BERT target model")
        parser.add_argument("--target_dataset",
                            default="mnli",
                            type=str,
                            help="Dataset Name")
        parser.add_argument("--fuzz",
                            default=0,
                            type=int,
                            help="Word Pruning Value")
        parser.add_argument("--top_k_words",
                            default=1000000,
                            type=int,
                            help="Top K Words")
        
        parser.add_argument("--budget",
                            type=int,
                            # required=True,
                            default=15000,
                            help="Number of Budget Limit")

        parser.add_argument("--test_dataset",
                            default="ag",
                            type=str,
                            help="所测试数据集的名称")
        parser.add_argument("--test_len",
                            default=1000,
                            type=int,
                            help="数据集测试长度")


    args = parser.parse_args()
    # ['infersent', 'esim', 'bert']
    #TODO: 这里根据不同的数据集选择不同的参数 不同的模型 不同的数据输入 不同的n_class 这里需要配置自己的路径
    model_dic = {
        "wordLSTM":"lstm",
        "bert":"bert",
        "wordCNN":"cnn",
        "esim":"esim"
    }
    dataset_info_dic = {
        "mnli": {
            "dataset_path": "/mnt/1TB-SSD-DATA/sy/hard_label/data/mnli",
            "target_model_path": "/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/mnli".format(model_dic[args.target_model]),
        },
        "snli": {
            "dataset_path": "/mnt/1TB-SSD-DATA/sy/hard_label/data/snli",
            "target_model_path": "/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/snli".format(model_dic[args.target_model]),
        },
        "mnli_matched": {
            "dataset_path": "/mnt/1TB-SSD-DATA/sy/hard_label/data/mnli_matched",
            # "target_model_path": "/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/mnli_matched".format(model_dic[args.target_model]),
            "target_model_path": "/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/mnli".format(
                model_dic[args.target_model]),
        },
        "mnli_mismatched":{
            "dataset_path":"/mnt/1TB-SSD-DATA/sy/hard_label/data/mnli_mismatched",
            # "target_model_path":"/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/mnli_mismatched".format(model_dic[args.target_model]),
            "target_model_path":"/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/mnli".format(model_dic[args.target_model]),
        }
    }
    args.target_model_path = dataset_info_dic[args.test_dataset]["target_model_path"]
    args.dataset_path = dataset_info_dic[args.test_dataset]["dataset_path"]


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    data = read_data(args.dataset_path, data_size=args.data_size, target_model=args.target_model)
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'esim':
        model = NLI_infer_ESIM(args.target_model_path,
                               args.word_embeddings_path,
                               batch_size=args.batch_size)
    elif args.target_model == 'infersent':
        model = NLI_infer_InferSent(args.target_model_path,
                                    args.word_embeddings_path,
                                    data=data,
                                    batch_size=args.batch_size)
    else:
        model = NLI_infer_BERT(args.target_model_path)
    predictor = model.text_pred
    print("Model built!")


    idx2word = {}
    word2idx = {}
    sim_lis=[]

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        with open(args.counter_fitting_cos_sim_path, "rb") as fp:
            sim_lis = pickle.load(fp)
    else:
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]

                embeddings.append(embedding)

        embeddings = np.array(embeddings,dtype='float64')
        embeddings = embeddings[:30000]


        print(embeddings.T.shape)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        cos_sim = np.dot(embeddings, embeddings.T)

    print("Cos sim import finished!")
    # if True 方便看代码
    if True:
        use = USE(args.USE_cache_path)

        orig_failures = 0.
        adv_failures = 0.
        avg=0.
        tot = 0
        elapsed_times = []
        changed_rates = []
        nums_queries = []
        orig_texts = []
        orig_premises=[]
        adv_texts = []
        true_labels = []
        new_labels = []
        wrds=[]
        s_queries=[]
        f_queries=[]
        success=[]
        results=[]
        fails=[]
        final_sims = []
        random_sims = []
        random_changed_rates = []
        log_dir = "nli_results_hard_label/"+args.target_model+"/"+args.test_dataset
        res_dir = "nli_results_hard_label/"+args.target_model+"/"+args.test_dataset
        log_file = "nli_results_hard_label/"+args.target_model+"/"+args.test_dataset+"/log.txt"
        result_file = "nli_results_hard_label/"+args.target_model+"/"+args.test_dataset+"/results_final.csv"
        process_file = os.path.join(log_dir,"sampled_process_log.txt")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        stop_words_set = criteria.get_stopwords()
        print('Start attacking!')
    res = []
    for idx, premise in enumerate(data['premises']):

        if idx % 20 == 0:
            print(str(idx)+" Samples Done. success num:{} . changed_rates:{}. final_sims:{}. qrs:{}. elapsed_time:{}.".format(len(success),np.mean(changed_rates),
                                                                                                            np.mean(final_sims),np.mean(nums_queries),
                                                                                                            np.mean(elapsed_times)))

        hypothese, true_label = data['hypotheses'][idx], data['labels'][idx]
        start_time = time.time()
        new_text, num_changed, random_changed, orig_label, \
        new_label, num_queries, sim, random_sim = mvgtext_attack(args.fuzz,args.top_k_words,
                                            idx,hypothese, premise, true_label, predictor, stop_words_set,
                                            word2idx, idx2word, sim_lis , sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,myargs=args,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size,embed_func = args.counter_fitting_embeddings_path,budget=args.budget,
                                            )
        end_time = time.time()
        elapsed_time = end_time - start_time

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)

        if true_label != new_label:
            adv_failures += 1
        changed_rate = 1.0 * num_changed / len(hypothese)
        random_changed_rate = 1.0 * random_changed / len(hypothese)
        if true_label == orig_label and true_label != new_label:
            print(orig_label,true_label,new_label)
            temp=[]
            s_queries.append(num_queries)
            success.append(idx)
            changed_rates.append(changed_rate)
            orig_texts.append(' '.join(hypothese))
            orig_premises.append(' '.join(premise))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)
            random_changed_rates.append(random_changed_rate)
            random_sims.append(random_sim)
            if type(sim) == type([]):
                sim = sim[0]
            final_sims.append(sim)
            temp.append(idx)
            temp.append(orig_label)
            temp.append(new_label)
            temp.append(' '.join(hypothese))
            temp.append(new_text)
            temp.append(num_queries)
            temp.append(random_sim)
            temp.append(sim)
            temp.append(changed_rate * 100)
            temp.append(random_changed_rate * 100)
            results.append(temp)
            elapsed_times.append(elapsed_time)
            print("Attacked: "+str(idx),"\tqrs",num_queries,"\tsim: ",sim,"\tnum_changed:",num_changed,"\telapsed_time:",elapsed_time)

    message =  'original accuracy: {:.3f}%, adv accuracy: {:.3f}%, random avg  change: {:.3f}% ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}, random_sims: {:.3f}, final_sims : {:.3f} \n'.format(
                                                                     (1-orig_failures/args.data_size)*100,
                                                                     (1-adv_failures/args.data_size)*100,
                                                                     np.mean(random_changed_rates)*100,
                                                                     np.mean(changed_rates)*100,
                                                                     np.mean(nums_queries),
                                                                     np.mean(random_sims),
                                                                     np.mean(final_sims))
    print(message)
    

    log=open(log_file,'a')
    log.write(message)
    with open(result_file,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)

    if args.target_model == 'bert':
        labeldict = {0: "contradiction",
                     1: "entailment",
                     2: "neutral"}
    else:
        labeldict = {0: "entailment",
                     1: "neutral",
                     2: "contradiction"}

    with open(os.path.join(args.output_dir, args.target_model+'_'+args.test_dataset+'_adversaries.txt'), 'w') as ofile:
        for orig_premise, orig_hypothesis, adv_hypothesis, true_label, new_label \
                in zip(orig_premises, orig_texts, adv_texts, true_labels, new_labels):
            ofile.write('orig premise:\t{}\norig hypothesis ({}):\t{}\n'
                        'adv hypothesis ({}):\t{}\n\n'.format(orig_premise,
                                                              labeldict[true_label],
                                                              orig_hypothesis,
                                                              labeldict[int(new_label)],
                                                              adv_hypothesis))

if __name__ == "__main__":
    main()
