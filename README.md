# Cost-effective bidding under partially observable prices landscapes

Real-time bidding has become a major mean for online advertisement exchange. The goal of a real-time bidding strategy is to maximize the benefits for stakeholders, e.g. click-through rates or conversion rates. 
However, in practise, the optimal bidding strategy for real-time bidding is constrained by three aspects: cost-effectiveness, the dynamic nature of market prices, and the issue of information scarcity in bidding values. 
To address these challenges, we propose Imagine and Imitate Bidding (IIBidder), which includes Strategy Imitation and Imagination modules, to generate cost-effective bidding strategies under partially observable prices landscapes.  
Experimental results on the iPinYou and YOYI datasets demonstrate that IIBidder reduces investment costs, optimizes bidding strategies, and improves future market price predictions.

## Files and Folders

- **datasets**   the datasets to train and test. Please note that due to limited space, the real large datasets should be downloaded from certain websites, as the datasets.ipynb instructs

- **expert_traj**  the expert knowledge trajectory, to be loaded by ours.py

- **methods** includes baseline algorithms and ours. Please note that ???

- README.md this file

- **ablation_exp.py**    to do ablation study

- **compared_exp.py** ???

- **globals.py**  global variables

- **main.py**    main entrance of the experiments. to envoke run.py, post.py, and plot.py

- **plot.py**  read final.csc, plot the figures used for the paper to store in `figures` folder, and generate  .tex files for tables for the papers.

- **post.py**  post-process,  to put together all the intermediate results in to one `final.csv` file.

- **preprocess.py**   

- **requirements.txt**  for install the conda virtual env.

- **rtb_environment.py**  ????

## Usage

1. Install Python 3.9. For convenience, execute the following command.

```shell
pip install -r requirements.txt
```

Another more elegant way to reproduce the result, of course, is use conda virtual environment, as widely appreciated. Typically by the following command:

```shell
conda create -n ImagineRTB python=3.9
```

We are not going to discuss the details here.

2. Prepare Data. 

download the original datasets from [IPINYOU](https://contest.ipinyou.com/) and [YOYI](https://apex.sjtu.edu.cn/datasets/7),
After downloading, preprocess the datasets as follows: 
(1) Group the datasets by advertiser objects, resulting in nine sub-datasets. Split the datasets into training and testing sets in a 2:1 ratio. 


(2) Retain the original columns 'click' and 'bid_price', convert the discrete numerical values of 'click' into continuous values and use regression models to learn other feature information to fit 'click', generating a third column 'ctr' representing click-through rate


(3) Save the processed data as a .txt file format. 
Partially processed datasets will be provided as examples for this project.



for convenience, we provide a `datasets.ipynb` file under the `methods` folder. Users are encouraged to excute it instead for the above operations.

3. Train and evaluate model. You can adjust parameters in global.py and reproduce the experiment results as the following examples:

```python
python3 main.py
```

As a scheduler, `main.py` will envoke `run.py`, `post.py`, and `plot.py` one by one, the functions of which are introduced in the files & folders part. 

4. Check the results
- results are in .csv format at `results` folder, which are later combined together to a `final.csv` for plotting purpose.
- figures are at `figures` folder
- latex tables are at `tables` folder

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

This work is supported by National Natural Science Foundation of China (62272198) and by Guangdong Provincial Science and Technology Plan Project (No.2021B1111600001).
