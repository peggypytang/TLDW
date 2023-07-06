import numpy as np
import re, math
import pickle as pkl
from torch.nn import functional as F
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils
import time
from utils import PAD, UNK, END, START, UNK_TOKEN, PAD_TOKEN
from torch.nn.functional import softplus
import torchvision.models as models
import nltk
from torchvision import transforms
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer, CLIPTextModel
from nltk.tokenize import sent_tokenize
from GPO import GPO
from torch_geometric.nn import GAT, global_mean_pool

class HOTNet(nn.Module):
    def __init__(self, text_hidden_size, video_hidden_size, 
                max_summary_word, max_summary_pic):

        super(HOTNet, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_summary_word = max_summary_word
        self.max_summary_pic = max_summary_pic
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.featureextractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        self.cliptext = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.cliptext.eval()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        self.hidden_size = 512
        self.dropout = 0.1

        self.multimodal_gat1 = GAT(self.hidden_size, self.hidden_size, 1)

        self.multimodal_gat2 = GAT(self.hidden_size, self.hidden_size, 1)

        self.multimodal_gat3 = GAT(self.hidden_size, self.hidden_size, 1)

        self.sentence_encoder = GPO(self.hidden_size, self.hidden_size)

        self.sentence_decoder = nn.GRU(self.hidden_size, self.hidden_size, 1, batch_first=True, bidirectional=True)

        self.text_encoder = GPO(self.hidden_size, self.hidden_size)

        self.text_decoder = nn.GRU(self.hidden_size*2, self.hidden_size, 1, batch_first=True, bidirectional=True)

        self.text_mm_decoder = nn.GRU(self.hidden_size*2, self.hidden_size, 1, batch_first=True, bidirectional=True)

        self.scene_encoder = GPO(self.hidden_size, self.hidden_size)

        self.scene_decoder = nn.GRU(self.hidden_size, self.hidden_size, 1, batch_first=True, bidirectional=True)

        self.video_encoder = GPO(self.hidden_size, self.hidden_size)

        self.video_decoder = nn.GRU(self.hidden_size*2, self.hidden_size, 1, batch_first=True, bidirectional=True)

        self.video_mm_decoder = nn.GRU(self.hidden_size*2, self.hidden_size, 1, batch_first=True, bidirectional=True)

        self.gat_edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).cuda()
        self.gat_batch = torch.tensor([0, 0], dtype=torch.long).cuda()
        self.outputs2vocab = nn.Linear(self.hidden_size*2, 1)

        self.outputs2coverframe = nn.Linear(self.hidden_size*2, 1)
        

    def forward(self, input_text, input_video, input_scene, text_summary_length=10):

        batch_size = len(input_text)
        
        batch_sent = []
        batch_sent_num = []
        batch_sent_pad = []
        for text in input_text:
            text = text.replace("Please subscribe HERE http://bit.ly/1rbfUog", "")
            text = text.replace("#BBCNews", "")
            sents = sent_tokenize(text)
            batch_sent.append(sents)
            batch_sent_num.append(len(sents))

        for batch in batch_sent:
            batch += [' '] * (max(batch_sent_num) - len(batch))
            batch_sent_pad.append(batch)

        batch_sent_pad_t = [list(x) for x in zip(*batch_sent_pad)]
        sent_feature = []
        word_feature = []
        text_token_list = []
        pad_sent_len = []

        last_len = 0
        for sent in batch_sent_pad_t:
            text_token = self.tokenizer(sent , return_tensors = "pt", padding=True, truncation=True, max_length=50)
            text_token_list.append(text_token['input_ids'])
            word_emb = self.cliptext(text_token['input_ids'].cuda()).last_hidden_state
            word_feature.append(word_emb)
            sent_feature.append(self.sentence_encoder(word_emb, torch.tensor(word_emb.size(1)).expand(batch_size).cuda()))
            pad_sent_len.append(text_token['input_ids'].size(1)+last_len)
            last_len += text_token['input_ids'].size(1)


        text_token_list = torch.cat(text_token_list, dim=1)
        word_feature = torch.cat(word_feature, dim=1)
        sent_feature = torch.stack(sent_feature,dim=1)

        if batch_size > 1:
            input_scene = torch.nn.utils.rnn.pad_sequence(input_scene, batch_first=True)

        scene_frame_batch_list = []
        for scenes, video in zip(input_scene, input_video):
            scene_frame = []
            for i, scene in enumerate(scenes):
                if i == 0:
                    scene_frame.append(video[:int(scene),:,:,:])
                else:
                    if scene == 0:
                        scene_frame.append(torch.zeros(video[0].size()).expand(1,-1,-1,-1))
                    else:
                        scene_frame.append(video[int(scenes[i-1]):int(scene),:,:,:])

            scene_frame_batch_list.append(scene_frame)

        pad_scene_frame_len_nontrim = []
        scene_frame_pad_batch_list = []

        last_len = 0
        for i in range(input_scene.size(1)):
            scene_nonpad = []
            for j in range(len(scene_frame_batch_list)):
                scene_nonpad.append(scene_frame_batch_list[j][i])
            scene_pad = torch.nn.utils.rnn.pad_sequence(scene_nonpad, batch_first=True)
            pad_scene_frame_len_nontrim.append(scene_pad.size(1)+last_len)
            scene_frame_pad_batch_list.append(scene_pad)
            last_len += scene_pad.size(1)
        scene_frame_pad_batch_list = torch.cat(scene_frame_pad_batch_list,dim=1)
        scene_frame_pad_batch_list = scene_frame_pad_batch_list[:, :100,:,:,:] #max for cuda
        pad_scene_frame_len = [x for x in pad_scene_frame_len_nontrim if x < 100]
        
        if len(pad_scene_frame_len) < len(pad_scene_frame_len_nontrim):
            pad_scene_frame_len.append(100)
        pad_scene_frame_len = sorted(list(set(pad_scene_frame_len)))

        v_image_features = []

        for i, images in enumerate(scene_frame_pad_batch_list.permute(1,0,4,2,3)):

            image_batch = []
            for batch in images:
                image_batch.append(transforms.ToPILImage()(batch.squeeze_(0)))
            image_token = self.featureextractor(image_batch, return_tensors = "pt")
            v_image_features.append(self.clip.get_image_features(image_token['pixel_values'].cuda()))


        image_feature = torch.stack(v_image_features, dim=1)

        text_overall_feature = self.text_encoder(word_feature, torch.tensor(word_feature.size(1)).expand(batch_size).cuda())
        video_overall_feature = self.video_encoder(image_feature, torch.tensor(image_feature.size(1)).expand(batch_size).cuda())


        multimodal_feature = []

        for t, v in zip(text_overall_feature, video_overall_feature):
            mm = torch.stack((t,v))
            mmpool = global_mean_pool(self.multimodal_gat1(mm, self.gat_edge), self.gat_batch)
            multimodal_feature.append(mmpool)

        multimodal_feature = torch.stack(multimodal_feature)


        image_feature_scene = []
        scene_feature_list = []
        for i, pad in enumerate(pad_scene_frame_len):
            if i == 0:
                scene_feature = self.scene_encoder(image_feature[:,torch.arange(start=0, end=pad).type(torch.LongTensor) ,:], torch.tensor(image_feature[:,torch.arange(start=0, end=pad).type(torch.LongTensor) ,:].size(1)).expand(batch_size).cuda())
                scene_feature_list.append(scene_feature)
                multimodal_scene_feature = []

                for t_b, v_b in zip(sent_feature, scene_feature):
                    update_scene = []
                    for t in t_b:
                        mm = torch.stack((t,v_b))
                        _, mmpool = self.multimodal_gat2(mm, self.gat_edge)
                        update_scene.append(mmpool)

                    multimodal_scene_feature.append(torch.mean(torch.stack(update_scene), dim=0))

                multimodal_scene_feature = torch.stack(multimodal_scene_feature).unsqueeze(1)

                image_feature_scene.append(self.scene_decoder(image_feature[:,torch.arange(start=0, end=pad).type(torch.LongTensor) ,:], multimodal_scene_feature.expand(-1,2,-1).permute(1,0,2).contiguous())[0])
            else:
                scene_feature = self.scene_encoder(image_feature[:, torch.arange(start=pad_scene_frame_len[i-1], end=pad).type(torch.LongTensor) ,:], torch.tensor(image_feature[:, torch.arange(start=pad_scene_frame_len[i-1], end=pad).type(torch.LongTensor) ,:].size(1)).expand(batch_size).cuda())
                multimodal_scene_feature = []
                scene_feature_list.append(scene_feature)

                for t_b, v_b in zip(sent_feature, scene_feature):
                    update_scene = []
                    for t in t_b:
                        mm = torch.stack((t,v_b))
                        _, mmpool = self.multimodal_gat2(mm, self.gat_edge)
                        update_scene.append(mmpool)

                    multimodal_scene_feature.append(torch.mean(torch.stack(update_scene), dim=0))

                multimodal_scene_feature = torch.stack(multimodal_scene_feature).unsqueeze(1)

                image_feature_scene.append(self.scene_decoder(image_feature[:, torch.arange(start=pad_scene_frame_len[i-1], end=pad).type(torch.LongTensor) ,:], multimodal_scene_feature.expand(-1,2,-1).permute(1,0,2).contiguous())[0])

        del image_feature
        del multimodal_scene_feature

        image_feature_scene = torch.cat(image_feature_scene, dim=1)
        scene_feature_list = torch.stack(scene_feature_list, dim=1)
        word_feature_sent = []
        for i, pad in enumerate(pad_sent_len):
            if i == 0:

                multimodal_sent_feature = []
                for t_b, v_b in zip(sent_feature[:,i ,:], scene_feature_list):
                    update_sent = []
                    for v in v_b:
                        mm = torch.stack((t_b,v))
                        mmpool, _ = self.multimodal_gat3(mm, self.gat_edge)
                        update_sent.append(mmpool)

                    multimodal_sent_feature.append(torch.mean(torch.stack(update_sent), dim=0))
                
                multimodal_sent_feature = torch.stack(multimodal_sent_feature)
                word_feature_sent.append(self.sentence_decoder(word_feature[:,torch.arange(start=0, end=pad).type(torch.LongTensor) ,:], multimodal_sent_feature.unsqueeze(1).expand(-1,2,-1).permute(1,0,2).contiguous())[0])
            else:

                multimodal_sent_feature = []
                for t_b, v_b in zip(sent_feature[:,i ,:], scene_feature_list):
                    update_sent = []
                    for v in v_b:
                        mm = torch.stack((t_b,v))
                        mmpool, _ = self.multimodal_gat3(mm, self.gat_edge)
                        update_sent.append(mmpool)
                        
                    multimodal_sent_feature.append(torch.mean(torch.stack(update_sent), dim=0))

                multimodal_sent_feature = torch.stack(multimodal_sent_feature)
                word_feature_sent.append(self.sentence_decoder(word_feature[:, torch.arange(start=pad_sent_len[i-1], end=pad).type(torch.LongTensor) ,:], multimodal_sent_feature.unsqueeze(1).expand(-1,2,-1).permute(1,0,2).contiguous())[0])


        del word_feature
        del scene_feature_list
        del multimodal_sent_feature

        word_feature_sent = torch.cat(word_feature_sent, dim=1)

        word_feature_text = self.text_decoder(word_feature_sent, text_overall_feature.unsqueeze(1).expand(-1,2,-1).permute(1,0,2).contiguous())[0]



        text_multimodal_z = self.text_mm_decoder(word_feature_text, multimodal_feature.expand(-1,2,-1).permute(1,0,2).contiguous())[0]

        text_b,text_s,_ = text_multimodal_z.size()

        text_logp = self.outputs2vocab(text_multimodal_z.reshape(-1, text_multimodal_z.size(2)))

        text_logp = text_logp.view(text_b, text_s, 1)
        

        image_feature_video = self.video_decoder(image_feature_scene, video_overall_feature.unsqueeze(1).expand(-1,2,-1).permute(1,0,2).contiguous())[0]


        video_multimodal_z = self.video_mm_decoder(image_feature_video, multimodal_feature.expand(-1,2,-1).permute(1,0,2).contiguous())[0]

        video_b,video_s,_ = video_multimodal_z.size()
        video_logp = self.outputs2coverframe(video_multimodal_z.reshape(-1, video_multimodal_z.size(2)))
        video_logp = video_logp.view(video_b, video_s, 1)


        output_video_summaries = []
        output_video_summaries_pos = []

        for image, summary in zip(scene_frame_pad_batch_list, video_logp):
            rank = torch.argsort(summary, dim=0, descending=True)
            output_video_summaries_pos.append(rank[0])
            output_video_summaries.append(image[int(rank[0])])
        
        output_text_summaries = []
        output_text_summaries_pos = []

        for text, summary in zip(text_token_list, text_logp):
            pos = []
            rank = torch.argsort(summary, dim=0, descending=True)
            text_id = []

            filtered_rank = []
            for i in rank:
                if i < len(text) and text[int(i)] < 49406:
                    filtered_rank.append(int(i))
                
                if len(filtered_rank) > self.max_summary_word:
                    break

            for i in sorted(filtered_rank):
                text_id.append(text[int(i)])
                pos.append(int(i))

            output_text_summaries.append(self.tokenizer.decode(text_id))
            output_text_summaries_pos.append(pos)
                    

        return output_text_summaries, output_text_summaries_pos, text_logp, output_video_summaries, output_video_summaries_pos, video_logp
