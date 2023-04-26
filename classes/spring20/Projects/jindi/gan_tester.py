from clf import Net
from data_loader import get_loader
import torch
import json
import os
from PIL import Image
from torchvision import transforms as T

'''
Generate the targeted labels

'''
def create_labels(device, c_org, c_dim=14, selected_attrs=None):
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        c_trg = c_org.clone()
        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            c_trg[:, i] = 1
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0
        else:
            c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

        c_trg_list.append(c_trg.to(device))
    return c_trg_list

def target_labels(data_loader, selected_attrs, device):
    dic = {}
    
    for i, (x_real, c_org) in enumerate(data_loader):
        x_real = x_real.to(device)
        c_trg_list = create_labels(device, c_org, 14, selected_attrs)

        temp = []
        
        temp.append(c_org.tolist())
        for j in range(len(c_trg_list)):
            temp.append(c_trg_list[j].tolist())

        dic[i] = temp

    return dic

'''
Return the predictions of classifier

'''
def tester(device):
    pre = []
    inputs = []

    transform = []    
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(178))
    transform.append(T.Resize(128))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    net = Net()
    ckp = 'ckp/imgclf_checkpoint10000.pth'
    net.load_state_dict(torch.load(ckp))
    net.to(device)

    imgs = []

    folder = "stargan_celeba/results"
    for i in range(0, 20000):
        for t in range(15):
            image = '%s-%simages.jpg'%(str(i), str(t))
            fn = os.path.join("stargan_celeba/results", image)
            img = Image.open(fn)
            imgs.append(img)
            img = transform(img) 
            inputs.append(img)  

    testloader = torch.utils.data.DataLoader(inputs, shuffle=False)
    dataiter = iter(testloader)
    dataiter.to(device)

    for data in tqdm(testloader, ncols=100): 
        image = data
        output = net(image)
        y = torch.zeros(1, 14)
        x = torch.ones(1, 14)
        output = torch.where(output > 0, x, y)
        pre.append(output)

    return pre
                

            


    

    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    celeba_image_dir = 'data/celeba/images'
    attr_path = 'data/celeba/list_attr_celeba.txt'
    selected_attrs = ["Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Eyeglasses", "Gray_Hair", 
                 "Male", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Hat", "Young"]
    celeba_crop_size = 178
    image_size = 128
    batch_size = 1
    mode = 'test'
    num_workers = 1
    celeba_loader = get_loader(celeba_image_dir, attr_path, selected_attrs,
                                celeba_crop_size, image_size, batch_size,
                                'CelebA', mode, num_workers)


    # dic = target_labels(celeba_loader, selected_attrs, device)
    # with open('truth.json', 'w') as f:
    #     json.dump(dic, f)

    '''
    The truth.json file here stores the generated image and their targeted labels
    '''
    # with open('truth.json', 'r') as f:
    #     truth = json.load(f)

    pre = tester(device)

    print('----')