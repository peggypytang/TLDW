import sys
import numpy as np
import os, shutil
import re
import codecs
import os, pickle as pkl
from util_dataset import EXMSMOWithSceneDataset
from HOTNet import HOTNet
import codecs
import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim
from torch import save
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils import make_vocab, make_embedding, convert_word2id, to_var, idx2word
from transformers import BertTokenizer
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer
from torchvision import transforms
import os
import torch.nn.functional as F
import torch.nn as nn
import gensim
import torch.nn.utils.rnn as rnn_utils
from multiprocessing import cpu_count
import time
import torch, gc
from collections import OrderedDict, defaultdict
from utils import PAD, UNK, END, START
import json
from torch.nn.functional import softplus
import torchvision.models as models
from datasets import load_dataset
import cv2 ,ot
import ssl
from model_generator import GeneTransformer
gc.collect()


ssl._create_default_https_context = ssl._create_unverified_context
BERT_NUM_TOKEN = 30522
torch.manual_seed(12345)


class TextCoverageLoss:
    # Depending on how many words are used a large fraction of the last X summaries
    def __init__(self, device="cuda", costmatrix_filename="COST_MATRIX_bert.pickle"):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_bytes = 2**31 - 1

        bytes_in = bytearray(0)
        input_size = os.path.getsize(costmatrix_filename)
        with open(costmatrix_filename, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)

            self.COST_MATRIX = pkl.loads(bytes_in)

    def score(self, summaries, bodies):
        scores = []
         # Avoid changing p and q outside of this function
        with torch.no_grad():
            for i in range(len(summaries)):

                summary = summaries[i]
                doc = bodies[i]
                if len(summary)==0:
                    score = 1
                else:
                    
                    summary_token = self.tokenizer.encode(summary) 
                    body_token = self.tokenizer.encode(doc)

                    summary_bow = construct_BOW(summary_token)
                    body_bow = construct_BOW(body_token)

                    score = sparse_ot(summary_bow, body_bow, self.COST_MATRIX) 

                scores.append(score)
        
        return sum(scores)/len(scores)


class MmCoverageLoss:
    def __init__(self, device="cuda"):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.featureextractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        self.clip.cuda()
        COSTMATRIX_DIM = 512
        self.COST_MATRIX = torch.ones(COSTMATRIX_DIM, COSTMATRIX_DIM) -  torch.eye(COSTMATRIX_DIM)
        self.COST_MATRIX = self.COST_MATRIX/COSTMATRIX_DIM


    def score(self, text_summaries, video_summaries, texts, videos):
        scores = []
        with torch.no_grad():
            for v, t in zip(video_summaries, text_summaries):
                i = transforms.ToPILImage()(v.permute(2,0,1).squeeze_(0))
                text_t = self.tokenizer(t, return_tensors = "pt", padding=True, truncation=True)
                image_t = self.featureextractor(i, return_tensors = "pt")
                text_f = self.clip.get_text_features(text_t['input_ids'].cuda())
                image_f = self.clip.get_image_features(image_t['pixel_values'].cuda())

                score = sparse_ot(text_f.squeeze(0).cpu().detach().numpy(), image_f.squeeze(0).cpu().detach().numpy(), self.COST_MATRIX.numpy()) 
                scores.append(score)

        return sum(scores)/len(scores)

class MmAlignmentLoss:
    def __init__(self, device="cuda"):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.featureextractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        self.clip.cuda()

        self.cosloss = nn.CosineEmbeddingLoss()

    def score(self, text_summaries, video_summaries):
        scores = []

        text_t = self.tokenizer(text_summaries, return_tensors = "pt", padding=True, truncation=True)

        image_batch = []
        for batch in video_summaries:
            image_batch.append(transforms.ToPILImage()(batch.permute(2,0,1).squeeze_(0)))

        image_t = self.featureextractor(image_batch, return_tensors = "pt")
        text_f = self.clip.get_text_features(text_t['input_ids'].cuda())
        image_f = self.clip.get_image_features(image_t['pixel_values'].cuda())

        scores.append(self.cosloss(text_f, image_f, Variable(torch.ones(text_f.size()[0]).cuda())))

        return scores[0]

def VideoCoverageLoss(summaries, bodies):
    scores = []
    # Avoid changing p and q outside of this function
    
    for summary, video in zip(summaries, bodies):
        video = np.mean(np.array(video), axis=0).astype(np.float32)
        summary = summary.detach().numpy().astype(np.float32)

        video_bw = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        summary_bw = cv2.cvtColor(summary, cv2.COLOR_BGR2GRAY)

        score = 1.0 / 0.001
        try:
            score = cv2.EMD(summary_bw,video_bw,cv2.DIST_L2)[0]
        except:
            print("VideoCoverageLoss cannot compute")
        scores.append(score)
        
    return sum(scores)/len(scores)
    
def save_log(log_input):
    file_name = MODEL_PATH + '/log.txt'
    p = log_input
    c = """text_file = open(file_name, "a+");text_file.write(p);text_file.close()""" 
    exec(c)


def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'validation'] + (['test'] if args.test else [])
    MODEL_PATH = args.save_model_path
    dataset_folder = args.dataset_folder
    

    params = dict(
        max_summary_word=args.max_summary_word,
        max_summary_pic=args.max_summary_pic,
        text_hidden_size=args.text_hidden_size,
        video_hidden_size=args.video_hidden_size,
        
    )


    max_bytes = 2**31 - 1

    def collate_func(inps):
        return [a for a in inps]

    train_dataset = EXMSMOWithSceneDataset('train', dataset_folder)
    val_dataset = EXMSMOWithSceneDataset('val', dataset_folder)

    print("Train Dataset size:", len(train_dataset))
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), drop_last=True, collate_fn=collate_func)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=collate_func)


    model = HOTNet(**params)

    
    if torch.cuda.is_available():
        model = model.cuda()

    if args.resume_training:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH,args.model_name)))
        model.train()


    print(model)


    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    def loss_fn(output_text_summaries, output_video_summaries, texts, videos, textCoverageLoss, mmCoverageLoss):

        textcoverage_loss = textCoverageLoss.score(output_text_summaries, texts)
        videocoverage_loss = VideoCoverageLoss(output_video_summaries, videos)
        mmcoverage_loss = mmCoverageLoss.score(output_text_summaries, output_video_summaries)
        fluency_loss, _ = fluencyLoss.score(output_text_summaries, output_video_summaries, descriptions, videos)

        return textcoverage_loss, videocoverage_loss , mmcoverage_loss, sum(fluency_loss)/len(fluency_loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    fluency_news_model_file = os.path.join("models", "gpt2_copier23.bin")

    textCoverageLoss = TextCoverageLoss()
    mmCoverageLoss = MmAlignmentLoss()
    fluencyLoss = GeneTransformer(max_output_length=args.max_summary_word, device="cuda", starter_model=fluency_news_model_file)
    
    for epoch in range(args.epochs):
        
        for split in splits:
            print("split", split)
            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
                data_loader = train_data_loader
            else:
                model.eval()
                data_loader = val_data_loader
            
            for iteration, data in enumerate(data_loader):
                gc.collect()
                torch.cuda.empty_cache()
                file_id = []
                descriptions = []
                videos = []
                titles = []
                scenes = []

                for d in data:
                    file_id.append(d[0])
                    descriptions.append(d[1])
                    videos.append(d[2])
                    titles.append(d[3])
                    scenes.append(d[6])

                batch_size = len(data)

                output_text_summaries, output_text_summaries_pos, text_logp, output_video_summaries, output_video_summaries_pos, video_logp = model(descriptions, videos, scenes)

                # loss calculation
                textcoverage_loss, videocoverage_loss, mmalignment_loss, fluency_loss = loss_fn(output_text_summaries, output_video_summaries, descriptions, videos, textCoverageLoss, mmCoverageLoss)

                loss = textcoverage_loss + 0.001 * videocoverage_loss + mmalignment_loss + fluency_loss
                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # bookkeepeing
                tracker['LOSS'] = torch.cat((tracker['LOSS'], loss.data.view(1, -1)), dim=0)

                if args.tensorboard_logging:
                    writer.add_scalar("%s/Loss" % split.upper(), loss.item(), epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/Text Coverage Loss" % split.upper(), textcoverage_loss,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/Video Coverage Loss" % split.upper(), videocoverage_loss,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/MM Alignment Loss" % split.upper(), mmalignment_loss,
                                      epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, Text-Coverage-Loss %9.4f, Video-Coverage-Loss %9.4f, MM-Alignment-Loss %9.4f"
                          % (split.upper(), iteration, len(data_loader)-1, loss.item(), textcoverage_loss, videocoverage_loss, mmalignment_loss.item()/batch_size))

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    #tracker['target_sents'] += idx2word(answer, i2w=datasets['train'].get_i2w(),pad_idx=PAD)
                    tracker['target_sents'] += idx2word(answer, i2w=self.id2word,pad_idx=PAD)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            print("%s Epoch %02d/%i, Mean LOSS %9.4f" % (split.upper(), epoch, args.epochs, tracker['LOSS'].mean()))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/LOSS" % split.upper(), torch.mean(tracker['LOSS']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json' % epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i-%9f.ckpt" % (epoch,tracker['LOSS'].mean()))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)



def sparse_ot(weights1, weights2, M):
    """ Compute Wasserstein distances"""
    
    weights1 = weights1/weights1.sum()
    weights2 = weights2/weights2.sum()
    
    active1 = np.where(weights1)[0]
    active2 = np.where(weights2)[0]
    
    weights_1_active = weights1[active1]
    weights_2_active = weights2[active2]
    try1 = M[active1][:,active2]
    M_reduced = np.ascontiguousarray(M[active1][:,active2])
    
    return ot.emd2(weights_1_active,weights_2_active,M_reduced)

def construct_BOW(tokens):
    bag_vector = np.zeros(BERT_NUM_TOKEN)        
    for token in tokens:            
        bag_vector[token] += 1                            
    return bag_vector/len(tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--max_sequence_length', type=int, default=900)
    parser.add_argument('--max_article_length', type=int, default=5)
    parser.add_argument('--max_summary_pic', type=int, default=1)
    parser.add_argument('--max_summary_word', type=int, default=12)

    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=100000)
    parser.add_argument('-bs', '--batch_size', type=int, default=2)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)

    parser.add_argument('-ths', '--text_hidden_size', type=int, default=128)
    parser.add_argument('-vhs', '--video_hidden_size', type=int, default=128)
    parser.add_argument('-nah', '--num_attention_head', type=int, default=2)
    parser.add_argument('-nl', '--num_layers', type=int, default=2)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='multimodal_model')
    parser.add_argument('--dataset_folder', type=str, help='folder of dataset')
    parser.add_argument("--resume_training", type=bool, default=False) 
    parser.add_argument("--model_name", type=str) 
    args = parser.parse_args()


    main(args)



