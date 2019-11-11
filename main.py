from magnet import *
from data_load import *
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

dataset = MagDataset(root_dir='./data')

dataset.size = dataset.len()
train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                        [0.8*dataset.size, 0.2*dataset.size])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=2)


def train_model(model, criterion, optimizer, scheduler, device, num_epochs=10):
    since = time.time()
    best_loss = 0.0
    if model != 0:
        best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:
            running_loss = 0.0

            # Iterate over data.
            for inputs, amplified_frame in dataloaders[phase]:
                inputs = inputs.to(device)
                amplified_frame = amplified_frame.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, amplified_frame)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0002, momentum=0.9)
device = torch.device("cuda:0")
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
net.to(device)
best_model = train_model(model=net, criterion=criterion, optimizer=optimizer,
            scheduler=exp_lr_scheduler, device=device)
save(best_model, 'best_mode.pt')
