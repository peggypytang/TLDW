# TLDW

This code is for paper [TLDW: Extreme Multimodal Summarisation of News Videos], IEEE Transactions on Circuits and Systems for Video Technology 2023

### To train the model:
> python train_HOTNet.py --dataset_folder /home/XMSMO_News --save_model_path [model_folder] 


### To test the model:
> python decode_HOTNet.py --dataset_folder /home/XMSMO_News --path=[decoded_model_folder] --test --model_dir=[model_folder] --model_name=[model_name]
