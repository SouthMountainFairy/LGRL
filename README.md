# LGRL: Local-Global Representation Learning for On-the-Fly FG-SBIR
-----
LGRL@IEEE Transactions on Big Data（TBD）.
Paper Download Address [PDF](https://ieeexplore.ieee.org/document/10409584 "悬停显示")

Abstract
---
On-the-fly Fine-grained sketch-based image retrieval (On-the-fly FG-SBIR) framework aim to break the barriers that sketch drawing requires excellent skills and is time-consuming. Considering such problems, a partial sketch with fewer strokes contains only the little local information, and the drawing process may show great difference among users, resulting in poor performance at the early retrieval. In this study, we developed a local-global representation learning (LGRL) method, in which we learn the representations for both the local and global regions of the partial sketch and its target photos. Specifically, we first designed a triplet network to learn the joint embedding space shared between the local and global regions of the entire sketch and its corresponding region of the photo. Then, we divided each partial sketch in the sketch-drawing episode into several local regions; Another learnable module following the triplet network was designed to learn the representations for the local regions of the partial sketch. Finally, by combining both the local and global regions of the sketches and photos, the final distance was determined. In the experiments, our method outperformed state-of-the-art baseline methods in terms of early retrieval efficiency on two publicly sketch-retrieval datasets and the practice test.

Architecture
---
![figure3](https://github.com/SouthMountainFairy/LGRL/blob/main/figure3.png)
An overview of our approach. (a) Stage 1: a conventional FG-SBIR framework with two independent channels; (b) Stage 2: Our proposed LGRL for on-the-fly FG-SBIR problem. The locks signify that the weights are fixed during learning.

Datasets and Preprocessing
---



Training and Testing
---



Citation
---
If you find this article useful in your research, please consider citing:
```
@ARTICLE{lgrl2024,
  author={Dai, Dawei and Liu, Yingge and Li, Yutang and Fu, Shiyu and Xia, Shuyin and Wang, Guoyin},
  journal={IEEE Transactions on Big Data}, 
  title={LGRL: Local-Global Representation Learning for On-the-Fly FG-SBIR}, 
  year={2024},
  volume={10},
  number={4},
  pages={543-555},
  keywords={Representation learning;Image retrieval;Big Data;Fine-grained sketch-based image retrieval (FG-SBIR);representation learning;triplet loss},
  doi={10.1109/TBDATA.2024.3356393}}
```

