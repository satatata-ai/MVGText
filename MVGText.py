# -*-coding:utf-8 -*-
# @Author  : SATAT

import sys

sys.path.append("../../")  # 相对路径或绝对路径

import argparse
import os
import numpy as np
from pathlib import Path

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

import sys

csv.field_size_limit(sys.maxsize)

import tensorflow_hub as hub

import tensorflow.compat.v1 as tf

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


class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        
        # 所有实验使用相同的USE
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


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        if torch.cuda.is_available():
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()
        else:
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses)

        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):

        self.model.eval()
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)
        probs_all = []
        for input_ids, input_mask, segment_ids in dataloader:
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


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

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

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

    def transform_text(self, data, batch_size=32):
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

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
    batch_size = 16
    total_semantic_sims = np.array([])
    for i in range(0, len(new_texts), batch_size):
        batch = new_texts[i:i + batch_size]
        semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), batch)))[0]
        total_semantic_sims = np.concatenate((total_semantic_sims, semantic_sims))
    return total_semantic_sims


def get_attack_result(new_text, predictor, orig_label, batch_size):
    '''
        查看attack是否成功
        return: true 攻击成功
                false 攻击失败
    '''
    new_probs = predictor(new_text, batch_size=batch_size)
    pr = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
    return pr



# 换回，添加了success_dict 和 fail_list  
def back_to_ori_word(random_text, text_ls, qrs, success_attack_dict, fail_attack_list, budget, sim_score_window,
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
                    pr = get_attack_result([new_text], predictor, orig_label, batch_size)
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
                    pr = get_attack_result([new_text], predictor, orig_label, batch_size)
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
                    random_text[choices[i][0]] = text_ls[choices[i][0]] 
                

        if len(choices) == 0:
            break

    return random_text, qrs, success_attack_dict, fail_attack_list

# 预算限制待优化（与论文实验无关）
def is_budget(sim_score_window, sim_predictor,budget,synonyms_dict,batch_size,text_ls,success_attack_dict,fail_attack_list,changed_indices,random_sim,orig_label,predictor,qrs):
    '''
    每次换回必检查，每次qrs+1也要检查
    '''

    # 加权，求新的排序，获得新的update_attack_list，只参考前100名
    update_attack_list_sim_pert_weight = list(
        sorted(success_attack_dict.items(), key=lambda item: item[1], reverse=True))[:100]

    sim_pert_weight_dict = {}
    for one in update_attack_list_sim_pert_weight:
        change_one = 0
        text_one = one[0].split(" ")
        for i in range(len(text_ls)):
            if text_ls[i] != text_one[i]:
                change_one += 1
        # sort_value = one[1] / change_one
        # sim_pert_weight_dict[one[0]]=one[1]/(change_one)
        sim_pert_weight_dict[one[0]] = change_one - one[1]

    new_sample = list(sorted(sim_pert_weight_dict.items(), key=lambda item: item[1], reverse=False))[0]
    new_text = new_sample[0]
    new_sim = new_sample[1]

    best_attack = new_text.split(" ")
    best_sim = new_sim
    # # 对当前最佳样本，前theta个重要单词，优化
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
            # 生成可替换的sample
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
                if " ".join(i) not in success_attack_dict.keys() and " ".join(i) not in fail_attack_list:
                    
                    pr = get_attack_result([i], predictor, orig_label, batch_size)
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
                    if qrs >= budget:
                        break

        all_values_false = all(value == False for value in check.values())

        if all_values_false:
            break
        elif not get_new:
            break


    pr = get_attack_result([best_attack], predictor, orig_label, batch_size)
    best_sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]
    # qrs += 1
    print("-------------------单样本优化结束-------------------")
    print("样本总数：", len(success_attack_dict.keys()) + len(fail_attack_list), "查询总数：", qrs, "best sim：", best_sim)


    del success_attack_dict
    del fail_attack_list

    # return
    max_changes = 0
    for i in range(len(text_ls)):
        if text_ls[i] != best_attack[i]:
            max_changes += 1

    now_label = torch.argmax(predictor([best_attack]))


    random_changed = 0
    if len(changed_indices) == 0:

        random_changed = max_changes
    else:
        random_changed = len(changed_indices)

    return ' '.join(best_attack), max_changes, random_changed,orig_label,now_label, qrs, best_sim, random_sim

# 我的算法
def mvgtext_attack(
        fuzz_val, top_k_words, sample_index, text_ls,
        true_label, predictor, stop_words_set, word2idx, idx2word,
        cos_sim, sim_predictor=None, import_score_threshold=-1.,
        sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
        batch_size=32, embed_func='', budget=1000, myargs=None):
    # print(budget)
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, 0, orig_label, orig_label, 0, 0, 0
    else:
        # word2idx 构建
        word_idx_dict = {}
        with open(embed_func, 'r') as ifile:
            for index, line in enumerate(ifile):
                word = line.strip().split()[0]
                word_idx_dict[word] = index
        
        embed_file = open(embed_func)
        embed_content = embed_file.readlines()

        pos_ls = criteria.get_pos(text_ls)
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1


        words_perturb = []
        words_perturb_cixing = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]

        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos and len(text_ls[i]) > 2:
                    words_perturb.append((i, text_ls[i]))
                    words_perturb_cixing.append((i, text_ls[i], pos_ls[i]))

        random.shuffle(words_perturb)

        words_perturb = words_perturb[:top_k_words]

        # get words perturbed idx embed doc_idx.find synonyms and make a dict of synonyms of each word.
        words_perturb_idx = []
        words_perturb_embed = []
        words_perturb_doc_idx = []
        for idx, word in words_perturb:
            if word in word_idx_dict:
                words_perturb_doc_idx.append(idx)
                words_perturb_idx.append(word2idx[word])
                words_perturb_embed.append(
                    [float(num) for num in embed_content[word_idx_dict[word]].strip().split()[1:]])

        words_perturb_embed_matrix = np.asarray(words_perturb_embed)

       
        synonym_words, synonym_values = [], []
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp = []
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            temp = []
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
            pr = get_attack_result([random_text], predictor, orig_label, batch_size)
            qrs+=1
            th +=1
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
            pr = get_attack_result([random_text], predictor, orig_label, batch_size)
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
            pr = get_attack_result([random_text], predictor, orig_label, batch_size)
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
            # 找到了，先换回
            keys = success_attack_dict.keys()
            # print("初始化找到的对抗样本数量：", now_adv_n, "alpha=", alpha)
            for one in list(keys):
                new_text, qrs, success_attack_dict, fail_attack_list, = back_to_ori_word(
                    one.split(" "), text_ls, qrs, success_attack_dict,
                    fail_attack_list, budget, sim_score_window,
                    sim_predictor, predictor, orig_label, batch_size)
               
                if qrs >= (budget*0.8):
                    new_sim = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)[0]
                    return is_budget(sim_score_window, sim_predictor,budget,synonyms_dict,batch_size,text_ls,success_attack_dict,fail_attack_list,[],new_sim,orig_label,predictor,qrs)

                if ' '.join(new_text) not in fail_attack_list and ' '.join(new_text) not in success_attack_dict.keys():
                    pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                    qrs += 1
                    if np.sum(pr) > 0:
                        semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        success_attack_dict[" ".join(random_text)] = semantic_sims[0]
                    else:
                        fail_attack_list.append(" ".join(random_text))

                    if qrs >= (budget*0.8):
                        new_sim = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)[0]
                        return is_budget(sim_score_window, sim_predictor,budget,synonyms_dict,batch_size,text_ls,success_attack_dict,fail_attack_list,[],new_sim,orig_label,predictor,qrs)

            
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
            print("随机初始化成功，当前更新列表中最好的一个", best_attack, "best sim：", random_sim, "\t修改单词数:", num_changed)

            only_one = False
            if num_changed == 1:  
                only_one=True
               
                print("-----------最佳样本，只替换一词-----------------")
            else:
                # 开始迭代优化
                search_times = 0
                search_th = 40  # 超过阈值次相同，搜索结束
                gamma = 20  # 多步迭代
                # for t in trange(50):  # 100
                tt = 0
                search_sim_th = -2.0
                while search_times < search_th and tt < 20:

                    tt += 1  # 外循环次数
                   
                    best_adv_embed = []
                    for idx in words_perturb_doc_idx:
                        best_adv_embed.append(
                            [float(num) for num in embed_content[word_idx_dict[best_attack[idx]]].strip().split()[1:]])
                    best_adv_embed_matrix = np.asarray(best_adv_embed)
                    best_noise = best_adv_embed_matrix-words_perturb_embed_matrix
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

                        
                        u=np.random.uniform(0.5, 1.0)
                       
                        new_noise = u * best_noise + (1 - u) * current_noise
                        new_noise_d = new_noise - np.random.normal(loc=0.0, scale=1,size=new_noise.shape)


                        # 可修改词下标列表，降噪前
                        nonzero_ele = np.nonzero(np.linalg.norm(new_noise, axis=-1))[0].tolist()
                       

                        perturb_word_idx_list = nonzero_ele
                        for perturb_idx in range(len(perturb_word_idx_list)):
                            perturb_word_idx = perturb_word_idx_list[perturb_idx]
                            
                            perturb_target = words_perturb_embed_matrix[perturb_word_idx] + new_noise_d[perturb_word_idx]  
                            syn_feat_set = []
                            for syn in synonyms_all[perturb_word_idx][1]:
                                syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]].strip().split()[1:]]
                                syn_feat_set.append(syn_feat)

                           
                            perturb_syn_dist = np.sum((syn_feat_set - perturb_target) ** 2, axis=1)
                            perturb_syn_order = np.argsort(perturb_syn_dist)
                            replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]

                            update_text[synonyms_all[perturb_word_idx][0]] = replacement

                        
                        if ' '.join(update_text) not in fail_attack_list and ' '.join(update_text) not in success_attack_dict.keys():
                            now_sim = float(calc_sim(text_ls, [update_text], -1, sim_score_window, sim_predictor)[0])
                            if now_sim > search_sim_th:
                                pr = get_attack_result([update_text], predictor, orig_label, batch_size)
                                qrs += 1

                                if np.sum(pr) > 0:
                                    semantic_sims = calc_sim(text_ls, [update_text], -1, sim_score_window, sim_predictor)
                                    success_attack_dict[" ".join(update_text)] = semantic_sims[0]
                                   
                                    # 换回寻找
                                    new_text, qrs, success_attack_dict, fail_attack_list, = back_to_ori_word(
                                        update_text, text_ls, qrs, success_attack_dict, fail_attack_list, budget,sim_score_window,
                                        sim_predictor, predictor, orig_label, batch_size)

                                    if qrs >= (budget*0.8):
                                        return is_budget(sim_score_window, sim_predictor,budget,synonyms_dict,batch_size,text_ls,success_attack_dict,fail_attack_list, changed_indices, random_sim,orig_label, predictor, qrs)

                                    if ' '.join(new_text) not in fail_attack_list and ' '.join(
                                            new_text) not in success_attack_dict.keys():
                                        
                                        pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                                        qrs += 1
                                        if np.sum(pr) > 0:
                                            semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window,
                                                                     sim_predictor)
                                            success_attack_dict[" ".join(new_text)] = semantic_sims[0]
                                        else:
                                            fail_attack_list.append(" ".join(new_text))
                                        if qrs >= (budget*0.8):
                                            return is_budget(sim_score_window, sim_predictor,budget,synonyms_dict,batch_size,text_ls,success_attack_dict,fail_attack_list,
                                                         changed_indices, random_sim,orig_label, predictor, qrs)
                                else:
                                    fail_attack_list.append(" ".join(update_text))
                                    
                                if qrs >= (budget*0.8):
                                    return is_budget(sim_score_window, sim_predictor,budget,synonyms_dict,batch_size,text_ls,success_attack_dict,fail_attack_list,
                                                     changed_indices, random_sim, orig_label, predictor, qrs)
                          

                        # 结束寻优，更新样本---------------------------
                        new_update_attack_list = list(sorted(success_attack_dict.items(), key=lambda item: item[1], reverse=True))[:alpha]

                        if new_update_attack_list == old_update_attack_list:
                            search_times += 1
                           
                        else:
                            # print("前alpha不一样,当前search_times=", search_times)
                            search_times = 0
                            # print("search_times归零:", search_times)

                        # 如果最好的样本只替换了一个单词，特殊优化：
                        best_attack = new_update_attack_list[0][0].split(" ")
                        best_sim = new_update_attack_list[0][1]

                        # 加权，求新的排序，获得新的update_attack_list，只参考前100名
                        update_attack_list_sim_pert_weight = list(sorted(success_attack_dict.items(), key=lambda item: item[1], reverse=True))[:100]
                        search_sim_th = update_attack_list_sim_pert_weight[-1][1]
                        # print("search_sim_th:",search_sim_th)

                        sim_pert_weight_dict = {}
                        for one in update_attack_list_sim_pert_weight:
                            change_one = 0
                            text_one = one[0].split(" ")
                            for i in range(len(text_ls)):
                                if text_ls[i] != text_one[i]:
                                    change_one += 1
                            
                            sim_pert_weight_dict[one[0]]=change_one - one[1]

                        update_attack_list = list(sorted(sim_pert_weight_dict.items(), key=lambda item: item[1], reverse=False))[:alpha]


                    print("第",tt,"次迭代，内循环gamma结束，更新best attack：", best_attack, "更新best sim：", best_sim, "查询次数：", qrs)
                print("多样本搜索结束，外循环迭代次数：", tt)
            print('成功样本数：', len(success_attack_dict.keys()),'失败样本数：', len(fail_attack_list))
            print("样本总数：", len(success_attack_dict.keys()) + len(fail_attack_list), "查询总数：", qrs,"best sim：", best_sim)
            

            if qrs >= (budget*0.8):
                return is_budget(sim_score_window, sim_predictor,budget,synonyms_dict,batch_size,text_ls,success_attack_dict,fail_attack_list, changed_indices, random_sim,
                              orig_label, predictor, qrs)


            # 单样本寻优
            if not only_one:
                success_best = list(sorted(sim_pert_weight_dict.items(), key=lambda item: item[1], reverse=False))[0]
                success_best_text = success_best[0]

            else:
                # 单样本寻优
                success_best = list(sorted(success_attack_dict.items(), key=lambda item: item[1], reverse=True))[0]
                success_best_text = success_best[0]

            best_attack = success_best_text.split(" ")
           
            best_sim = success_attack_dict[success_best_text]
            
            choices = []
            check= dict()
            for i in range(len(text_ls)):
                if text_ls[i] != best_attack[i]:
                    now_best = best_attack[:]
                    now_best[i] = text_ls[i]
                    semantic_sims = calc_sim(text_ls, [now_best], -1, sim_score_window, sim_predictor)
                    choices.append((i, semantic_sims[0]))  # 优化顺序
                    check[str(i)]=True
            print("修改单词数：",len(choices))
           
            choices_list = choices[:]

            print("-------------------开始单样本优化！-------------------------")
            
            while True:
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
                    candi_samples_filter =replaced_txt
                    print("倒排sim找最大sim,candi_samples_filter中样本个数：",len(candi_samples_filter),"qrs:",qrs)

                    for i,sim in candi_samples_filter:
                        if " ".join(i) not in success_attack_dict.keys() and " ".join(i) not in fail_attack_list:
                           
                            pr = get_attack_result([i], predictor, orig_label, batch_size)
                            qrs += 1
                            if np.sum(pr) > 0:
                                success_attack_dict[" ".join(i)] = sim
                                if sim > best_sim:
                                    for x in range(len(text_ls)):
                                        if i[x] != best_attack[x]:
                                            check[str(x)]=False
                                            get_new = True

                                    best_attack = i
                                    best_sim = sim
                                    print("单样本优化更新啦！！！！当前sim:",best_sim)
                                break
                            else:
                                fail_attack_list.append(i)
                            if qrs >= (budget*0.8):
                                return is_budget(sim_score_window, sim_predictor,budget,synonyms_dict,batch_size,text_ls,success_attack_dict,fail_attack_list, changed_indices,
                                                 random_sim, orig_label, predictor, qrs)

                all_values_false = all(value == False for value in check.values())

                if all_values_false:
                    break
                elif not get_new:
                    break


            pr = get_attack_result([best_attack], predictor, orig_label, batch_size)
            # qrs += 1
            print("-------------------单样本优化结束-------------------")
            print("样本总数：", len(success_attack_dict.keys()) + len(fail_attack_list), "查询总数：", qrs, "best sim：", best_sim)


            del success_attack_dict
            del fail_attack_list

            # return
            max_changes = 0
            for i in range(len(text_ls)):
                if text_ls[i] != best_attack[i]:
                    max_changes += 1

            return ' '.join(best_attack), max_changes, len(changed_indices), \
                   orig_label, torch.argmax(predictor([best_attack])), qrs, best_sim, random_sim


def main():
    # if True 方便看代码
    if True:
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset_path",
                            type=str,
                            # required=True,
                            default="/mnt/1TB-SSD-DATA/sy/hard_label/data/mr",
                            help="Which dataset to attack.")
        parser.add_argument("--nclasses",
                            type=int,
                            default=2,
                            help="How many classes for classification.")
        parser.add_argument("--target_model",
                            type=str,
                            # required=True,
                            default="wordCNN",
                            choices=['wordLSTM', 'bert', 'wordCNN'],
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
                            default='mvgtext_adv_results' + version_name,
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
                            default="imdb_test",
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
                            # default=15000,
                            default=15000,  # 默认预算，
                            help="Number of Budget Limit")

        parser.add_argument("--test_dataset",
                            default="mr",
                            type=str,
                            help="所测试数据集的名称")
        parser.add_argument("--test_len",
                            default=30,
                            type=int,
                            help="数据集测试长度")

    args = parser.parse_args()
    # TODO: 这里根据不同的数据集选择不同的参数 不同的模型 不同的数据输入 不同的n_class 这里需要配置自己的路径
    model_dic = {
        "wordLSTM": "lstm",
        "bert": "bert",
        "wordCNN": "cnn"
    }
    dataset_info_dic = {
        "imdb": {
            "dataset_path": "/mnt/1TB-SSD-DATA/sy/hard_label/data/imdb",
            "target_model_path": "/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/imdb".format(model_dic[args.target_model]),
            "n_classes": 2
        },
        "yelp": {
            "dataset_path": "/mnt/1TB-SSD-DATA/sy/hard_label/data/yelp",
            "target_model_path": "/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/yelp".format(model_dic[args.target_model]),
            "n_classes": 2
        },
        "yahoo": {
            "dataset_path": "/mnt/1TB-SSD-DATA/sy/hard_label/data/yahoo",
            "target_model_path": "/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/yahoo".format(model_dic[args.target_model]),
            "n_classes": 10
        },
        "ag": {
            "dataset_path": "/mnt/1TB-SSD-DATA/sy/hard_label/data/ag",
            "target_model_path": "/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/ag".format(model_dic[args.target_model]),
            "n_classes": 4
        },
        "mr": {
            "dataset_path": "/mnt/1TB-SSD-DATA/sy/hard_label/data/mr",
            "target_model_path": "/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/mr".format(model_dic[args.target_model]),
            "n_classes": 2
        },
        "imdb_test": {
            "dataset_path": "/mnt/1TB-SSD-DATA/sy/hard_label/data/mr",
            "target_model_path": "/mnt/1TB-SSD-DATA/sy/hard_label/model/{}/mr".format(model_dic[args.target_model]),
            "n_classes": 2
        }
    }
    args.target_model_path = dataset_info_dic[args.test_dataset]["target_model_path"]
    args.dataset_path = dataset_info_dic[args.test_dataset]["dataset_path"]
    args.nclasses = dataset_info_dic[args.test_dataset]["n_classes"]

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    texts, labels = dataloader.read_corpus(args.dataset_path, csvf=False)
    data = list(zip(texts, labels))
    data = data[:args.data_size]
    print("Data import finished!")

    print("Building Model...")

    if args.target_model == 'wordLSTM':
        if torch.cuda.is_available():
            model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
            checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
            model.load_state_dict(checkpoint)
        else:
            model = Model(args.word_embeddings_path, nclasses=args.nclasses)
            checkpoint = torch.load(args.target_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        if torch.cuda.is_available():
            model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=150, cnn=True).cuda()
            checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
            model.load_state_dict(checkpoint)
        else:
            model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=150, cnn=True)
            checkpoint = torch.load(args.target_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
    predictor = model.text_pred
    print("Model built!")

    idx2word = {}
    word2idx = {}
    sim_lis = []

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

        embeddings = np.array(embeddings, dtype='float64')
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
        avg = 0.
        tot = 0
        elapsed_times = []
        changed_rates = []
        nums_queries = []
        orig_texts = []
        adv_texts = []
        true_labels = []
        new_labels = []
        wrds = []
        s_queries = []
        f_queries = []
        success = []
        results = []
        fails = []
        final_sims = []
        random_sims = []
        random_changed_rates = []
        all_cixing = []
        log_dir = "mvgtext_results_hard_label_" + version_name + "/" + args.target_model + "/" + args.test_dataset
        res_dir = "mvgtext_results_hard_label_" + version_name + "/" + args.target_model + "/" + args.test_dataset
        log_file = "mvgtext_results_hard_label_" + version_name + "/" + args.target_model + "/" + args.test_dataset + "/log.txt"
        result_file = "mvgtext_results_hard_label_" + version_name + "/" + args.target_model + "/" + args.test_dataset + "/results_final.csv"
        process_file = os.path.join(log_dir, "sampled_process_log.txt")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        stop_words_set = criteria.get_stopwords()
        print('Start attacking!')
    res = []
    for idx, (text, true_label) in enumerate(data[:args.test_len]):

        '''
        text = ["i'm", 'convinced', 'i', 'could', 'keep', 'a', 'family', 
        'of', 'five', 'blind', ',', 'crippled', ',', 'amish', 'people',
        'alive', 'in', 'this', 'situation', 'better', 'than', 'these', 
        'british', 'soldiers', 'do', 'at', 'keeping', 'themselves', 'kicking']
        true_label = 0
        '''
        if idx % 20 == 0:
            print(str(
                idx) + " Samples Done. success num:{} . changed_rates:{}. final_sims:{}. qrs:{}. elapsed_time:{}.".format(
                len(success), np.mean(changed_rates),
                np.mean(final_sims), np.mean(nums_queries),
                np.mean(elapsed_times)))
            # print(final_sims)
            # with open(os.path.join(log_dir, 'sampled_processd.txt'), 'a') as ofile:
            #     ofile.write('{} Samples Done \t final_sims :{}\tnums_queries :{}\t avg changed rate: {}%\n'.format(str(idx),np.mean(final_sims),
            #                                                                                                       np.mean(nums_queries),np.mean(changed_rates)*100))
        # print(" ".join(text))
        start_time = time.time()
        # print("当前样本是："+str(idx))
        new_text, num_changed, random_changed, orig_label, \
        new_label, num_queries, sim, random_sim = mvgtext_attack(args.fuzz, args.top_k_words,
                                                                  idx, text, true_label, predictor, stop_words_set,
                                                                  word2idx, idx2word, sim_lis, sim_predictor=use,
                                                                  sim_score_threshold=args.sim_score_threshold,
                                                                  myargs=args,
                                                                  import_score_threshold=args.import_score_threshold,
                                                                  sim_score_window=args.sim_score_window,
                                                                  synonym_num=args.synonym_num,
                                                                  batch_size=args.batch_size,
                                                                  embed_func=args.counter_fitting_embeddings_path,
                                                                  budget=args.budget,
                                                                  )
        end_time = time.time()
        elapsed_time = end_time - start_time

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)

        if true_label != new_label:
            adv_failures += 1
        changed_rate = 1.0 * num_changed / len(text)
        random_changed_rate = 1.0 * random_changed / len(text)
        if true_label == orig_label and true_label != new_label:
            temp = []
            s_queries.append(num_queries)
            success.append(idx)
            changed_rates.append(changed_rate)
            orig_texts.append(' '.join(text))
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
            temp.append(' '.join(text))
            temp.append(new_text)
            temp.append(num_queries)
            temp.append(random_sim)
            temp.append(sim)
            temp.append(changed_rate * 100)
            temp.append(random_changed_rate * 100)
            results.append(temp)
            elapsed_times.append(elapsed_time)
            # 统计词性
            changed_tongji = 0
            cixing = []
            cixing.append(idx)
            for i in range(len(text)):
                adv_text = new_text.split(" ")
                if text[i] != adv_text[i]:
                    changed_tongji += 1
                    pos_ls = criteria.get_pos([adv_text[i]])
                    # print([text[i] + "->" + adv_text[i], pos_ls])  # pos存在('DET',)
                    pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
                    cixing.append([text[i] + "->" + adv_text[i], pos_ls[0]])

            cixing.insert(1, str(len(cixing) - 1))
            cixing.insert(2, "ori_text: " + ' '.join(text))
            cixing.insert(3, "num_changed: " + str(num_changed))
            cixing.insert(4, "adv_text: " + str(new_text))
            cixing.insert(5, "changed_tongji: " + str(changed_tongji))
            if changed_tongji != num_changed:
                cixing.insert(6, "not equal")
            else:
                cixing.insert(6, "equal")
            print(cixing)
            all_cixing.append(cixing)

            print("Attacked: " + str(idx), "\tqrs", num_queries, "\tsim: ", sim, "\tnum_changed:", num_changed,
                  "\telapsed_time:", elapsed_time)
        if true_label == orig_label and true_label == new_label:
            f_queries.append(num_queries)
            temp1 = []
            temp1.append(idx)
            temp1.append(' '.join(text))
            temp1.append(new_text)
            temp1.append(num_queries)
            fails.append(temp1)

    # joblib.dump(res, "kmeans_first_"+args.target_model+".pkl")
    message = 'original accuracy: {:.3f}%, adv accuracy: {:.3f}%, random avg  change: {:.3f}% ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}, random_sims: {:.3f}, final_sims : {:.3f} \n'.format(
        (1 - orig_failures / args.data_size) * 100,
        (1 - adv_failures / args.data_size) * 100,
        np.mean(random_changed_rates) * 100,
        np.mean(changed_rates) * 100,
        np.mean(nums_queries),
        np.mean(random_sims),
        np.mean(final_sims))
    print(message)

    log = open(log_file, 'a')
    log.write(message)
    with open(result_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)

    with open(res_dir + "/tongjicixing.csv", 'w') as csvfile2:
        csvwriter2 = csv.writer(csvfile2)
        csvwriter2.writerows(all_cixing)

    with open(os.path.join(args.output_dir, args.target_model + '_' + args.test_dataset + '_adversaries.txt'),
              'w') as ofile:
        for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
            ofile.write(
                'orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))


if __name__ == "__main__":
    main()
