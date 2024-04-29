# Cost-effective bidding under partially observable prices landscapes

Real-time bidding has become a major mean for online advertisement exchange. The goal of a real-time bidding strategy is to maximize the benefits for stakeholders, e.g. click-through rates or conversion rates. 
However, in practise, the optimal bidding strategy for real-time bidding is constrained by three aspects: cost-effectiveness, the dynamic nature of market prices, and the issue of information scarcity in bidding values. 
To address these challenges, we propose Imagine and Imitate Bidding (IIBidder), which includes Strategy Imitation and Imagination modules, to generate cost-effective bidding strategies under partially observable prices landscapes.  
Experimental results on the iPinYou and YOYI datasets demonstrate that IIBidder reduces investment costs, optimizes bidding strategies, and improves future market price predictions.

## Files & Folders

- **datasets**   the datasets to train and test. Please note that due to limited space, the real large datasets should be downloaded from certain websites, as the `datasets.ipynb` instructs the way of processing raw datasets. 

- **expert_traj**  the expert knowledge trajectory, to be loaded by `ours.py`.

- **methods** includes baseline algorithms and ours. Please note that `IIBidder_agent` is our algorithm. DRLB, Uniform, Normal, Lin, and Gamma are included in the compare experiment. AC_GAIL_agent, PPO_GAIL_agent, and PPO_agent are included in the ablation experiment.

- **README.md** this file

- **ablation_exp.py**    to do ablation study

- **compared_exp.py**    to do comparison among different baselines (DRLB, Uniform, Normal, Lin, and Gamma)

- **globals.py**  global variables

- **main.py**    main entrance of the experiments. to envoke `run.py`, `post.py`, and `plot.py`

- **plot.py**  read `final.csv`, plot the figures as .pdf used for the paper to store in `figures` folder, and generate  latex tables as .tex files for the papers.

- **post.py**  post-process,  to put together all the intermediate results in to one `final.csv` file.

- **preprocess.py**   read data from 'datasets' folder, and preprocess for compare experiment and ablation experiment.

- **requirements.txt**  for installation of the conda virtual env.

- **rtb_environment.py**  create a bidding environment for agent

## Usage

1. Install Python 3.9. For convenience, execute the following command.

```shell
pip install -r requirements.txt
```

Another more elegant way to reproduce the result, of course, is use conda virtual environment, as widely appreciated. Typically by entering the following command before the above pip installation:

```shell
conda create -n ImagineRTB python=3.9
```

We are not going to discuss the details here.

2. Prepare Data. 

Download the original datasets from [IPINYOU](https://contest.ipinyou.com/) and [YOYI](https://apex.sjtu.edu.cn/datasets/7). Process the raw dataset according to the instruction in `datasets.ipynb` under the `datasets` folder. 

Considering the dataset size and space limitations, we provide a mini IPINYOU datasets `train.theta_2997.txt` and `test.theta_2997.txt` for users to directly invoke and experiment with. As suggested, users could actually skip this data preparation step to the 3rd step and run the `main.py` to get the picture before diving themselves into the complete datasets.


3. Train and evaluate model. You can adjust parameters in global.py and reproduce the experiment results as the following examples:

```python
python3 main.py
```

As a scheduler, `main.py` will envoke `run.py`, `post.py`, and `plot.py` one by one, the functions of which are introduced in the **Files & Folders** part. 

4. Check the results
- results are in .csv format at `results` folder, which are later combined together to a `final.csv` for plotting purpose.
- figures are at `figures` folder
- latex tables are at `tables` folder

## Citation

If you find this repo useful, please cite our paper.

```
 Luo, X.; Chen, Y.; Zhuo, S.; Lu, J.; Chen, Z.; Li, L.; Tian, J.; Ye, X.; Tang, Y. Imagine and Imitate: Cost-Effective Bidding under Partially Observable Price Landscapes. Big Data Cogn. Comput. 2024, 8, 46. https://doi.org/10.3390/ bdcc8050046
```

## Contact

If you have any questions or suggestions, feel free to contact:

- Xiaotong Luo <xiaotong01@stu2020.jnu.edu.cn>
- Yin Tang <ytang@jnu.edu.cn>

Or describe it in Issues.

## Acknowledgement

This work is supported by National Natural Science Foundation of China (62272198) and by Guangdong Provincial Science and Technology Plan Project (No.2021B1111600001).
