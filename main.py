from run import run
from post import post
from plot import plot_and_tex
import pandas as pd
from globals import plot_param, post_param

def main():
    # 输出中间结果到 result/
    run()
    # 输出单独final.csv
    post()
    # 输出图和表到 figures 
    data = pd.read_csv(post_param['output_file'])
    abl = plot_param['ablation']
    comp = plot_param['compared']
    abl_category = ['boxplot'] 
    comp_category = ['cloud_rain', 'heatmap', 'budget_lineplot', 'mask_lineplot']
    plot_and_tex(data, abl, abl_category)
    plot_and_tex(data, comp, comp_category)

if __name__ == '__main__':
    main()
