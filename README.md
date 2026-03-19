# Not All Sequences Need Augmentation: Retrieval-Augmented Diffusion with Contrastive Learning for Sequential Recommendation (RADRec)


This is the Pytorch implementation for the paper.


## Implementation
### Requirements
```
python>=3.9
Pytorch >= 1.12.0
torchvision==0.13.0
torchaudio==0.12.0
numpy==1.24.4
scipy==1.6.0
pandas==2.2.3
```
### Datasets
Four public datasets are used in our experiments:

- [Beauty](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/All_Beauty_5.json.gz): the All Beauty subset from the Amazon Review Data (2018) collection.
- [Sports](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Sports_and_Outdoors_5.json.gz): the Sports and Outdoors subset from the Amazon Review Data (2018) collection.
- [Yelp](https://business.yelp.com/data/resources/open-dataset/): the Yelp Open Dataset for business recommendation and user review interactions.
- [ML-1M](https://grouplens.org/datasets/movielens/1m/): the MovieLens 1M benchmark released by GroupLens.



### Evaluate RADRec
Here are the trained models for the Beauty datasets, stored in the `./output` folder. <br>
You can evaluate a checkpoint in the `./output` folder directly on the test set by running the following command:

```
python main.py --data_name Beauty --eval_only --checkpoint_path ./output/RADRec-Beauty.pt
```

### Train RADRec

```
python main.py --data_name Beauty --model_idx 0 --build_entropy_cache_only --entropy_pretrained_path ./pretrained/Beauty-0.pt
python main.py --data_name Beauty --model_idx 0 --entropy_pretrained_path ./pretrained/Beauty-0.pt
```

We will be releasing the complete code for the paper RADRec, so stay tuned!


