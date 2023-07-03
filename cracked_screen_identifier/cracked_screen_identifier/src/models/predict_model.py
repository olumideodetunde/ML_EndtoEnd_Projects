model.load_state_dict(torch.load(f'{log_dir}/modelnet.pth'))
model.eval()
correct = 0
total = 0
predictions = []
with torch.no_grad():  
    for data, label in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        predictions.extend(predicted.tolist())
        
        #Print the image and the prediction
        for img, pred in zip(data, predicted):
            img = img.permute(1, 2, 0) #Reshape the image from (C, H, W) to (H, W, C)
            plt.imshow(img)
            plt.title(f"Predicted: {pred}")
            plt.show()
        
    print('Test Accuracy of the model: {} %'.format(100 * correct / total))
    print('Predictions: ', predictions)