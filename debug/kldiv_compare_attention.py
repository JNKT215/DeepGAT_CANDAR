import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


def calc_kldiv(a_vec, b_vec):
    return np.sum([a * np.log(a/b) for a, b in zip(a_vec, b_vec)])

def calc_v_attention_kldiv(L2_attention,max_L_attention):
    output = []
    for v_attention_l2, v_attention_max_l in zip(L2_attention, max_L_attention):
        epsilon_vec = np.array([1e-5 for _ in range(v_attention_l2.shape[0])])
        v_attention_l2 += epsilon_vec
        v_attention_max_l += epsilon_vec
        kl_div = calc_kldiv(v_attention_l2, v_attention_max_l)
        output.append(kl_div)
    return output

def visualize_attention_kldiv(kl_divs,save_dir,output_name,outlier):
    divs  = [v for n, v in kl_divs.items()]
    names = [n for n, v in kl_divs.items()]

    fig, ax = plt.subplots()
    ax.boxplot(divs, sym=outlier)
    ax.set_xticklabels(names)
    # ax.set_ylabel("KLD(Att_l=2,Att_l=9)")
    plt.tick_params(labelsize=18)
    plt.ylim([-0.01,0.16])
    ax.set_yticks([0,0.05,0.10,0.15])
    plt.savefig(f'{save_dir}{output_name}.png')

def load_attention(args):
    DeepGAT_L2_Attention = np.load(args.DeepGAT_L2_att, allow_pickle=True)
    DeepGAT_max_L_Attention = np.load(args.DeepGAT_max_L_att, allow_pickle=True)
    GAT_L2_Attention = np.load(args.GAT_L2_att, allow_pickle=True)
    GAT_max_L_Attention = np.load(args.GAT_max_L_att, allow_pickle=True)
    return DeepGAT_L2_Attention,DeepGAT_max_L_Attention,GAT_L2_Attention,GAT_max_L_Attention
    
    

if __name__ == "__main__":
    save_dir = "DeepGAT/output/kldiv/"
    os.makedirs(save_dir, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='None')
    parser.add_argument('--att_type', type=str, default='None')
    parser.add_argument('--DeepGAT_L2_att', type=str, default='None')
    parser.add_argument('--DeepGAT_max_L_att', type=str, default='None')
    parser.add_argument('--GAT_L2_att', type=str, default='None')
    parser.add_argument('--GAT_max_L_att', type=str, default='None')
    parser.add_argument('--output_name', type=str, default='None')
    parser.add_argument('--outlier', type=str, default='')
    args = parser.parse_args()

    
    DeepGAT_L2_Attention,DeepGAT_max_L_Attention,GAT_L2_Attention,GAT_max_L_Attention = load_attention(args)
    
    
    kl_divs = {}
    kl_divs[f"DeepGAT ({args.att_type[1:]})"] = calc_v_attention_kldiv(DeepGAT_L2_Attention,DeepGAT_max_L_Attention)
    kl_divs[f"GAT ({args.att_type[1:]})"]     = calc_v_attention_kldiv(GAT_L2_Attention,GAT_max_L_Attention)
    print(f"dataset:{args.name}")
    visualize_attention_kldiv(kl_divs,save_dir,args.output_name,args.outlier)
    
    