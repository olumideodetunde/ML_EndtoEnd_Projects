#%%
import torch
import matplotlib.pyplot as plt
from model_architecture import ModelNet
from dataloader import PhoneDataset
from train_model import Trainer

log_dir = "logs/20230707-221539-Experiment 2 - Baseline Model"
model = torch.load(f'{log_dir}/modelnet.pth')
model.eval()
correct = 0
total = 0
predictions = []

test_data_dir = "data/processed/test"
test_csv = "data/processed/test/label.csv"
test_dataset = PhoneDataset(test_data_dir, test_csv)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True)

print(model)
#%%
with torch.no_grad():  
    for image, label in test_loader:
        image = image.float()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        predictions.extend(predicted.tolist())
        
        #Print the image and the prediction
        for img, pred in zip(image, predicted):
            img = img.long().permute(1,2,0)
            plt.imshow(img)
            plt.title(f"Predicted: {pred}")
            plt.show()
    print('Test Accuracy of the model: {} %'.format(100 * correct / total))
    print('Predictions: ', predictions)