from magnet import *
from data_load import *
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import copy


def train_model(model, criterion, optimizer, scheduler, device,
                dataloaders, dataset_sizes, num_epochs=10):
    since = time.time()
    best_loss = 0.0
    if model != 0:
        best_model_wts = copy.deepcopy(model.state_dict())

    trace_loss = {'train':[],'val':[]}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:
            filehandle = open(phase+'_origin.txt', 'a+')
            running_loss = 0.0
            epoch_loss = 0

            # Iterate over data.
            for i, inputs in enumerate(dataloaders[phase]):
                img_a = inputs['frameA'].to(device, dtype=torch.float)
                img_b = inputs['frameB'].to(device, dtype=torch.float)
                img_c = inputs['frameC'].to(device, dtype=torch.float)
                amplified = inputs['amplified'].to(device, dtype=torch.float)
                amplification_factor = inputs['amplification_factor'].to(device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(img_a, img_b, amplification_factor)
                    text_c, shape_c = model.encoder(img_c)
                    loss = criterion(outputs, amplified) + criterion(model.text_a, text_c) \
                           + criterion(model.shape_a, shape_c)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                if i % 10 == 9:
                    print('images num:{} running: {:.4f}'.format((i+1)*4, running_loss / 10.0))
                    epoch_loss += running_loss / dataset_sizes[phase]
                    trace_loss[phase].append(float(running_loss) / 10.0)
                    filehandle.write('{:.5f}\n'.format(float(running_loss) / 10.0))
                    running_loss = 0
            if phase == 'train':
                scheduler.step()

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            filehandle.close()

            # deep copy the model
            if phase == 'val' and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

tra_val_dataset = MagDataset(root_dir='C:/Wenyu/data', transform=ToTensor())

dataset_sizes = {'train': int(0.8*len(tra_val_dataset)), 'val': int(0.2*len(tra_val_dataset))}
trainsubset, testsubset = torch.utils.data.random_split(tra_val_dataset,
                                        [dataset_sizes['train'], dataset_sizes['val']])
trainset = MagSubset(trainsubset)
testset = MagSubset(testsubset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True)
dataloaders = {'train': trainloader, 'val': testloader}

net = origin_Net()

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.00001)
device = torch.device("cuda:0")
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
net.to(device)
best_model = train_model(model=net, criterion=criterion, optimizer=optimizer,
            scheduler=exp_lr_scheduler, device=device, dataloaders=dataloaders, dataset_sizes=dataset_sizes)
torch.save(best_model, 'best_mode.pt')
places = ['Berlin', 'Cape Town', 'Sydney', 'Moscow']
