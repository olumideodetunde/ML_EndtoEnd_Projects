#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import PhoneDataset

class Predictor:
    
    def __init__(self, test_data_dir, test_data_labelcsv, log_dir):
        self.test_data_dir = test_data_dir
        self.test_data_labelcsv = test_data_labelcsv
        self.log_dir = log_dir
        self.predictions = []

    def load_model(self):
        self.model = torch.load(f'{self.log_dir}/modelnet.pth')
        return self.model

    def prepare_test_data(self):
        self.test_dataset = PhoneDataset(self.test_data_dir, self.test_data_labelcsv)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=2, shuffle=True)
        return self.test_dataset, self.test_loader
    
    def make_prediction(self):
        results = []
        with torch.no_grad():
            self.model.eval()
            for i in self.test_dataset:
                image = i[0]
                image = image.float()
                label = image[1]
                inference = self.model(image)
                image = image.long().permute(2,1,0)
                _, predicted = torch.max(inference.data, 1)
                mapped_prediction = ['cracked' if predicted == 1 else "Perfect"] #prediction decoded
                results.append((image, label, mapped_prediction[0]))
        return results
                     
    def save_predicted_images(self):
        predictions = self.make_prediction()
        num_images, num_rows, num_cols = len(predictions), 8, 7
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
        fig.tight_layout()
        for i, (image, label, prediction) in enumerate(predictions):
            row, col  = i // num_cols,i % num_cols
            axes[row, col].imshow(image)
            axes[row, col].set_title(f"Pred: {prediction}")
            axes[row, col].axis('off')
        for j in range(num_images, num_rows * num_cols):
            row, col = j // num_cols, j % num_cols
            axes[row, col].axis('off')
        plt.savefig("reports/figures/prediction.png")
        plt.show()

    def obtain_prediction(self):
        self.load_model()
        self.prepare_test_data()
        self.make_prediction()
        self.save_predicted_images()
        
def main(data_dir, label, log):
    predict = Predictor(data_dir, label, log)
    predict.obtain_prediction()

if __name__ == "__main__":
    main(data_dir="data/processed/test", label="data/processed/test/label.csv", 
         log="logs/20230707-221539-Experiment 2 - Baseline Model")