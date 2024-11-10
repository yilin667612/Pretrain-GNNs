import torch 

def compute_hom(root, dataset_name):
    dataset = torch.load(root + dataset_name)

    hit = 0
    for i in range(len(dataset[0]['edge_index'][0])):
        if dataset[0]['x'][dataset[0]['edge_index'][0][i]][0] == dataset[0]['x'][dataset[0]['edge_index'][1][i]][0]:
            hit = hit + 1

    hom_ratio = 1.0 * hit / (len(dataset[0]['edge_index'][0]))

    return hom_ratio

if __name__ == "__main__":
    root_unsupervised = '/home/dyl/Graph-code/pretrain-gnns/bio/dataset/unsupervised'
    dataset_name = '/processed/geometric_data_processed.pt'
    
    result = compute_hom(root_unsupervised, dataset_name)
    print(f"hom_ratio: {result}")
