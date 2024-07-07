import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image

args = sys.argv

#Variables
pretrained = False
image_path = ''
model_path = './classifier.pth'
classes = []

#Hyper parameters

bsize = 4
loss_rate = 0.001
momentum = 0.9
output_path = "./classifier.pth"
train_iterations = 2

if len(args) == 1:
    print('please provide proper arguments refer to github page for more information')
    sys.exit()

if '-train' in args:
    pretrained = False
    iters = args[args.index('-train') + 1]
    if iters.isnumeric():
        train_iterations = eval(iters)
    else:
        print("Train Iterations Must be a valid Number")
        sys.exit()
    classes = ['airplane' ,'automobile','bird' ,'cat' ,'deer' ,'dog' ,'frog' ,'horse' ,'ship','truck']

if '-image' in args:
    pretrained = True
    image_path = args[args.index('-image') + 1]

if '-classes' in args:
    pretrained = False
    classes = args[args.index('-classes') + 1].split(',')
elif pretrained:
    classes = ['airplane' ,'automobile','bird' ,'cat' ,'deer' ,'dog' ,'frog' ,'horse' ,'ship','truck']


#Utility Functions

def imshow(img ,isTensor, label="input"):
    if(isTensor):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(label)
        plt.show()
    else:
        plt.imshow(img)
        plt.title(label)
        plt.show()


#Defining Transform to resize image and convert to Pytorch Tensor
tranform = transforms.Compose([transforms.Resize((32,32)) ,transforms.ToTensor() , transforms.Normalize((0.5 ,0.5,0.5) , (0.5,0.5,0.5) )])

#Loading the dataset if -train flag is set to true
if not pretrained:
    training_set = torchvision.datasets.CIFAR10('./data/train' , train=True , transform=tranform , download=True)
    validation_set = torchvision.datasets.CIFAR10('./data/test' , train=False , transform=tranform , download=True)

    #Create Data Loaders

    train_dataloader = torch.utils.data.DataLoader(training_set , batch_size = bsize , shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(validation_set , batch_size = bsize , shuffle=False)

    print('training set: ',len(training_set))
    print('validation set: ' ,len(validation_set))


#Printing Parameters for training
if not pretrained:
    print()
    print('Batch Size: ',bsize)
    print('Loss Rate: ',loss_rate)
    print('Momentum: ',momentum)
    print('Training Iterations: ',train_iterations)
    print('Output Path: ',output_path)
    print()

#Definining the Model

class ImageClassifier(nn.Module):

    def __init__(self):
        super(ImageClassifier , self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=6 , kernel_size=5) #3 input channel R G B , output to 6 , with kernel size of 5*5
        self.pool = nn.MaxPool2d(kernel_size=2 , stride=2) # Find maximum value in 2*2 square on output tensor
        self.conv2 = nn.Conv2d(in_channels=6 , out_channels=16 , kernel_size=5) # antoher convolution layer 

        self.fc1 = nn.Linear(in_features=16*5*5 , out_features= 128) # in feature is 16 * 5 *5  where 16 is output of conv2 with a feature map of 5 * 5 
        self.fc2 = nn.Linear(in_features=128 , out_features= 84)
        self.fc3 = nn.Linear(in_features=84 , out_features= len(classes))
    
    def forward(self ,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1 , 16*5*5) #flatten image to 1d into proper format alternative torch.flatten(x ,1) , used x.view for clarity purpose

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

#Instanciating Model

net = ImageClassifier()

if pretrained:
    net.load_state_dict(torch.load(model_path))

#defining loss function and optimizer for training mode
if not pretrained:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters() , lr = loss_rate , momentum= momentum) #Scotastic Gradient Descent 

#Implementing Training Loop

if not pretrained:
    for epoch in range(train_iterations):

        running_loss = 0.0

        for i , data in enumerate(train_dataloader):

            #Parse the data
            image , labels = data

            optimizer.zero_grad() #reset gradient values

            outputs = net(image)
            
            #calculate loss
            loss = criterion(outputs , labels)
            loss.backward()

            optimizer.step()

            #print stats
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('\r [',epoch +1 ,', ',i +1,'] running loss = ',running_loss * (1/100),'%')
                running_loss = 0.0

    torch.save(net.state_dict() , output_path) #save the model
    print('Training finished')

if pretrained:
    img = Image.open(image_path)
    image = tranform(img)

    output = net(image)

    _,prediction = torch.max(output , 1)

    print()
    print("The predicted object is " , classes[prediction.item()])
    imshow(img , False , label=classes[prediction.item()])

#Testing

if not pretrained:
    #testing

    dataiter = iter(test_dataloader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images) , True)
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(bsize))) #The correct image type

    #Load model
    network = ImageClassifier()
    network.load_state_dict(torch.load(output_path))

    #get and print predicted outputs
    outputs = net(images)

    _,predicted = torch.max(outputs ,1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

    #Test for general Acurracy
    correct = 0
    total = 0
    # don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    #Test for accuracy for particular object types
    # dicts to count the predications for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # no gradients needed
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')