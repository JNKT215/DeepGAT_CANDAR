# Deep Graph Attention Networks (CANDARW 2024)

This repository is the implementation of [Deep Graph Attention Networks (DeepGAT)](https://arxiv.org/pdf/2410.15640).

> Jun Kato, Airi Mita, Keita Gobara, Akihiro Inokuchi, Deep Graph Attention Networks, The Twelfth International Symposium on Computing and Networking (CANDARW 2024).


## Requirements
```bash
torch
torch_geometric
hydra
mlflow
tqdm
```

## Guide to experimental replication

CS dataset with DeepGAT
```bash
#best (num_layer=2, att_type=DP)
python3 train_coauthor.py key=DeepGAT_CANDAR_cs_tuned_YDP DeepGAT_CANDAR_cs_tuned_YDP.num_layer=2
#best (num_layer=2, att_type=SD)
python3 train_coauthor.py key=DeepGAT_CANDAR_cs_tuned_YSD DeepGAT_CANDAR_cs_tuned_YSD.num_layer=2
#max (num_layer=15, att_type=DP)
python3 train_coauthor.py key=DeepGAT_CANDAR_cs_tuned_YDP DeepGAT_CANDAR_cs_tuned_YDP.num_layer=15
#max (num_layer=15, att_type=SD)
python3 train_coauthor.py key=DeepGAT_CANDAR_cs_tuned_YSD DeepGAT_CANDAR_cs_tuned_YSD.num_layer=15
```

Physics dataset with DeepGAT
```bash
#best (num_layer=4, att_type=DP)
python3 train_coauthor.py key=DeepGAT_CANDAR_physics_tuned_YDP DeepGAT_CANDAR_physics_tuned_YDP.num_layer=4
#best (num_layer=5, att_type=SD)
python3 train_coauthor.py key=DeepGAT_CANDAR_physics_tuned_YSD DeepGAT_CANDAR_physics_tuned_YSD.num_layer=5
#max (num_layer=15, att_type=DP)
python3 train_coauthor.py key=DeepGAT_CANDAR_physics_tuned_YDP DeepGAT_CANDAR_physics_tuned_YDP.num_layer=15
#max (num_layer=15, att_type=SD)
python3 train_coauthor.py key=DeepGAT_CANDAR_physics_tuned_YSD DeepGAT_CANDAR_physics_tuned_YSD.num_layer=15
```

Flickr dataset with DeepGAT
```bash
#best (num_layer=2, att_type=DP)
python3 train_flickr.py key=DeepGAT_CANDAR_flickr_tuned_YDP DeepGAT_CANDAR_flickr_tuned_YDP.num_layer=2
#best (num_layer=4, att_type=SD)
python3 train_coauthor.py key=DeepGAT_CANDAR_flickr_tuned_YSD DeepGAT_CANDAR_flickr_tuned_YSD.num_layer=4
#max (num_layer=9, att_type=DP)
python3 train_flickr.py key=DeepGAT_CANDAR_flickr_tuned_YDP DeepGAT_CANDAR_flickr_tuned_YDP.num_layer=9
#max (num_layer=9, att_type=SD)
python3 train_coauthor.py key=DeepGAT_CANDAR_flickr_tuned_YSD DeepGAT_CANDAR_flickr_tuned_YSD.num_layer=9
```

PPI dataset with DeepGAT
```bash
#best (num_layer=7, att_type=DP)
python3 train_ppi.py key=DeepGAT_CANDAR_ppi_tuned_YDP DeepGAT_CANDAR_ppi_tuned_YDP.num_layer=7
#best (num_layer=8 or 9, att_type=SD)
python3 train_ppi.py key=DeepGAT_CANDAR_ppi_tuned_YSD DeepGAT_CANDAR_ppi_tuned_YSD.num_layer=8
#max (num_layer=9, att_type=DP)
python3 train_ppi.py key=DeepGAT_CANDAR_ppi_tuned_YDP DeepGAT_CANDAR_ppi_tuned_YDP.num_layer=9
#max (num_layer=9, att_type=SD)
python3 train_ppi.py key=DeepGAT_CANDAR_ppi_tuned_YSD DeepGAT_CANDAR_ppi_tuned_YSD.num_layer=9
```
If you need to know the parameters in detail, please check conf/config.yaml.