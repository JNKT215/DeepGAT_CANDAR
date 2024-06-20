import numpy as np
import argparse

def overlap_nodes_num(h_path,y_path,supervised_index_path):
    y = np.load(y_path)
    h = np.load(h_path)
    supervised_index = np.load(supervised_index_path)
    unsupervised_index = np.ones(len(y),dtype=bool)
    unsupervised_index[supervised_index] = False

    supervised_h = h[supervised_index]
    unsupervised_h = h[unsupervised_index]
    supervised_y = y[supervised_index]
    unsupervised_y = y[unsupervised_index]

    cnt = 0
    for u,u_y in zip(unsupervised_h,unsupervised_y):
        dist_temp,y_temp = [],[]
        for v,v_y in zip(supervised_h,supervised_y):
            dist = np.linalg.norm(v-u,ord=2)
            dist**=2
            dist_temp.append(dist)
            y_temp.append(v_y)
        dist_temp_np = np.array(dist_temp)
        y_temp_np = np.array(y_temp)
        
        v_y_index = np.argmin(dist_temp_np)
        u_y_pred = y_temp_np[v_y_index]
        if u_y != u_y_pred:
            cnt+=1
    print(f"overlaps_nodes_num:{cnt}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='None')
    parser.add_argument('--y_path', type=str, default='None')
    parser.add_argument('--supervised_index_path',type=str, default='None')
    parser.add_argument('--L0_h_path', type=str, default='None')
    parser.add_argument('--L1_h_path', type=str, default='None')
    parser.add_argument('--L2_h_path', type=str, default='None')
    parser.add_argument('--L3_h_path', type=str, default='None')
    parser.add_argument('--L4_h_path', type=str, default='None')
    parser.add_argument('--L5_h_path', type=str, default='None')
    parser.add_argument('--L6_h_path', type=str, default='None')
    parser.add_argument('--L7_h_path', type=str, default='None')
    parser.add_argument('--L8_h_path', type=str, default='None')
    parser.add_argument('--L9_h_path', type=str, default='None')
    parser.add_argument('--L10_h_path', type=str, default='None')
    parser.add_argument('--L11_h_path', type=str, default='None')
    parser.add_argument('--L12_h_path', type=str, default='None')
    parser.add_argument('--L13_h_path', type=str, default='None')
    parser.add_argument('--L14_h_path', type=str, default='None')
    parser.add_argument('--L15_h_path', type=str, default='None')
    args = parser.parse_args()

    L = [i for i in range(16)]
    h_paths =[
        args.L0_h_path,
        args.L1_h_path,
        args.L2_h_path,
        args.L3_h_path,
        args.L4_h_path,
        args.L5_h_path,
        args.L6_h_path,
        args.L7_h_path,
        args.L8_h_path,
        args.L9_h_path,
        args.L10_h_path,
        args.L11_h_path,
        args.L12_h_path,
        args.L13_h_path,
        args.L14_h_path,
        args.L15_h_path,
    ]
    
    print(f"dataset:{args.name}")
    for l,h_path in zip(L,h_paths):
        print(f":{l}層目")
        overlap_nodes_num(h_path,args.y_path,args.supervised_index_path)