from magnet import *
from data_load import *
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg.astype(int)
    tmp = np.transpose(npimg, (1, 2, 0))
    tmp = np.array(tmp, dtype=int)
    print(tmp)

    plt.imshow(tmp)
    plt.show()

class Mag_test(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, amplification_factor=20):
        self.img_name = glob.glob(os.path.join(root_dir, '*.jpg'))
        self.root_dir = root_dir
        self.transform = transform
        self.amplification_factor = amplification_factor

    def __len__(self):
        return len(self.img_name)-1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_a = Image.open(self.img_name[idx])
        img_b = Image.open(self.img_name[idx+1])
#        img_a = img_a.resize((384, 384))
#        img_b = img_b.resize((384, 384))

        sample = {'frameA': np.asarray(img_a), 'frameB': np.asarray(img_b),
                  'amplification_factor': self.amplification_factor}

        if self.transform:
            sample = self.transform(sample)

        return sample

device = torch.device('cpu')
root_dir='F:/heartrate/Video_Mag/data/'
test_dataset = Mag_test(root_dir='F:/heartrate/Video_Mag/data/', transform=ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                          shuffle=True)

PATH = './best_mode.pt'
model = origin_Net()
model.to(device)
model = torch.load(PATH, map_location=lambda storage, loc: storage)

with torch.no_grad():
    for i, data in enumerate(test_loader):
        img_a = data['frameA'].to(device, dtype=torch.float)
        img_b = data['frameB'].to(device, dtype=torch.float)
        amplification_factor = data['amplification_factor'].to(device, dtype=torch.float)
        outputs = model(img_a, img_b, amplification_factor)
        outputs = np.transpose(outputs[0].numpy(), (1, 2, 0))/2 + 0.5
        outputs = np.clip(outputs, 0, 1)
        plt.imsave(root_dir + 'output/' + str(i) + '.jpg', outputs)