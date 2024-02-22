# Cost-effective bidding under partially observable prices landscapes

Real-time bidding has become a major mean for online advertisement exchange. The goal of a real-time bidding strategy is to maximize the benefits for stakeholders, e.g. click-through rates or conversion rates. 
However, in practise, the optimal bidding strategy for real-time bidding is constrained by three aspects: cost-effectiveness, the dynamic nature of market prices, and the issue of information scarcity in bidding values. 
To address these challenges, we propose Imagine \& Imitate Bidding (\ours), which includes Strategy Imitation and Imagination modules, to generate cost-effective bidding strategies under partially observable prices landscapes.  
Experimental results on the iPinYou and YOYI datasets demonstrate that \ours~reduces investment costs, optimizes bidding strategies, and improves future market price predictions.

codes are to be released soon

:triangular_flag_on_post:**News** (2023.10) We add an implementation to [iTransformer](https://arxiv.org/abs/2310.06625), which is the state-of-the-art model for long-term forecasting. The official code and complete scripts of iTransformer can be found [here](https://github.com/thuml/iTransformer).

:triangular_flag_on_post:**News** (2023.09) We added a detailed [tutorial](https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb) for [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq) and this library, which is quite friendly to beginners of deep time series analysis.

## Leaderboard for Time Series Analysis

Till October 2023, the top three models for five different tasks are:

| Model&lt;br&gt;Ranking | Long-term&lt;br&gt;Forecasting | Short-term&lt;br&gt;Forecasting | Imputation | Classification | Anomaly&lt;br&gt;Detection |
| --- | --- | --- | --- | --- | --- |
| 🥇 1st | [iTransformer](https://arxiv.org/abs/2310.06625) | [TimesNet](https://arxiv.org/abs/2210.02186) | [TimesNet](https://arxiv.org/abs/2210.02186) | [TimesNet](https://arxiv.org/abs/2210.02186) | [TimesNet](https://arxiv.org/abs/2210.02186) |
| 🥈 2nd | [PatchTST](https://github.com/yuqinie98/PatchTST) | [Non-stationary&lt;br/&gt;Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary&lt;br/&gt;Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary&lt;br/&gt;Transformer](https://github.com/thuml/Nonstationary_Transformers) | [FEDformer](https://github.com/MAZiqing/FEDformer) |
| 🥉 3rd | [TimesNet](https://arxiv.org/abs/2210.02186) | [FEDformer](https://github.com/MAZiqing/FEDformer) | [Autoformer](https://github.com/thuml/Autoformer) | [Informer](https://github.com/zhouhaoyi/Informer2020) | [Autoformer](https://github.com/thuml/Autoformer) |

**Note: We will keep updating this leaderboard.** If you have proposed advanced and awesome models, you can send us your paper/code link or raise a pull request. We will add them to this repo and update the leaderboard as soon as possible.

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.

- [x] **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
- [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
- [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py).
- [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py).
- [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py).
- [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py).
- [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py).
- [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py).
- [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py).
- [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py).
- [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py).
- [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py).
- [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py).

See our latest paper [[TimesNet]](https://arxiv.org/abs/2210.02186) for the comprehensive benchmark. We will release a real-time updated online version soon.

**Newly added baselines.** We will add them to the leaderboard after a comprehensive evaluation.

- [x] **Koopa** - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors [[NeurIPS 2023]](https://arxiv.org/pdf/2305.18803.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Koopa.py).
- [x] **FreTS** - Frequency-domain MLPs are More Effective Learners in Time Series Forecasting [[NeurIPS 2023]](https://arxiv.org/pdf/2311.06184.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FreTS.py).
- [x] **TiDE** - Long-term Forecasting with TiDE: Time-series Dense Encoder [[arXiv 2023]](https://arxiv.org/pdf/2304.08424.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiDE.py).
- [x] **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/forum?id=zTQdHSQUQWc)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py).
- [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py).
- [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py).

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
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# classification
bash ./scripts/classification/TimesNet.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{luo2024cost,
  title={cost effective bidding},
  author={Xiaotong Luo, Yin Tang∗},
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


