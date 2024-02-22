# Cost-effective bidding under partially observable prices landscapes

Real-time bidding has become a major mean for online advertisement exchange. The goal of a real-time bidding strategy is to maximize the benefits for stakeholders, e.g. click-through rates or conversion rates. 
However, in practise, the optimal bidding strategy for real-time bidding is constrained by three aspects: cost-effectiveness, the dynamic nature of market prices, and the issue of information scarcity in bidding values. 
To address these challenges, we propose Imagine and Imitate Bidding (IIBidder), which includes Strategy Imitation and Imagination modules, to generate cost-effective bidding strategies under partially observable prices landscapes.  
Experimental results on the iPinYou and YOYI datasets demonstrate that IIBidder reduces investment costs, optimizes bidding strategies, and improves future market price predictions.

codes are to be released soon


## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. Here is a summary of supported datasets.

&lt;p align=&quot;center&quot;&gt;
&lt;img src=&quot;.\pic\dataset.png&quot; height = &quot;200&quot; alt=&quot;&quot; align=center /&gt;
&lt;/p&gt;

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
python3 main.py
```

4. Check the results
-  results are in .csv format at folder 
- figures are at folder
- latex tables are at folder


## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{luo2024cost,
  title={Cost-effective bidding under partially observable prices landscapes},
  author={Xiaotong Luo, Yin Tang},
  journal={},
  year={2024},
}
```

## Contact

If you have any questions or suggestions, feel free to contact:

- Xiaotong Luo()
- Yin Tang (ytang@jnu.edu.cn)


Or describe it in Issues.

## Acknowledgement

This work is supported by National Natural Science Foundation of China (62272198)


