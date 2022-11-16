import pickle
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Training for TransNet')
parser.add_argument('--model_name_list', '-net', type=list, default=['metric_vit','metric_viswin','metric_conv'])
parser.add_argument('--dataset', '-dt', type=str, default='kfw1')
parser.add_argument('--type', '-tp', type=list, default=['fd','fs','md','ms'])



name_ls = ['KT-ViT', 'KT-VST', 'KT-IResNet']
args = parser.parse_args()

for tp in args.type:
    roc_dicts = {}
    for model_name in args.model_name_list:
        svpth = "./results/{}/roc/{}".format(args.dataset,model_name)
        roc_pth = svpth + "/{}-roc_all.pkl".format(tp)
        with open(roc_pth, 'rb') as f:
            roc_all = pickle.load(f)
        roc_dicts[model_name] = roc_all


    plt.figure()
    plt.xlabel("FPR", fontsize = 20)
    plt.ylabel("TPR", fontsize = 20)
    # plt.title("ROC Curve -{}-{}".format(self.tp,self.cs), fontsize = 14)
    plt.plot([0, 1], [0, 1], color="navy", lw=4, linestyle="--")
    i = 0
    for model_name in roc_dicts:
        roc_all = roc_dicts[model_name]
        fpr = roc_all[0]
        tpr = roc_all[1]
        
        plt.plot(fpr, tpr, lw = 4,label = name_ls[i])
        i+=1
    # plt.plot(fpr, tpr, linewidth = 2, label = model_name)
    plt.legend(loc="lower right", fontsize = 20)
    svpth = "./results/{}/roc/".format(args.dataset)

    plt.savefig(svpth+"/ROC-Curve-{}-cs-all".format(tp))


    plt.close()