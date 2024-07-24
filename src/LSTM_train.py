import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from data_funcs import *

class VehicleCountLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(VehicleCountLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_vehicle_count_data(csv_files):
    all_vehicle_counts = []
    for csv_file in csv_files:
        with open(csv_file, 'r') as file_data:
            reader = csv.reader(file_data)
            next(reader)
            vehicle_counts = [int(row[1]) for row in reader]
            all_vehicle_counts.append(vehicle_counts)
    return all_vehicle_counts

def get_vehicle_detection_csv_files(data_dir):
    csv_files = []
    for file_data in os.listdir(data_dir):
        if file_data.endswith('.csv') and 'vehicle_detection_result_seed' in file_data:
            csv_files.append(os.path.join(data_dir, file_data))
    return csv_files

def create_sequences(data_list, window_size, prediction_offset):
    X, y = [], []
    for data_vid in data_list:
        for i in range(len(data_vid) - window_size - prediction_offset):
            X.append(data_vid[i:i + window_size])
            y.append(data_vid[i + window_size + prediction_offset])
    return np.array(X), np.array(y)

def calculate_loss(model, criterion, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)
    return loss.item()

def train_lstm(data_list, window_size, prediction_offset, data_dir, hidden_size = 4, num_layers = 1, batch_size = 32, num_epochs = 50, lr = 0.0001, seed = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess data
    normalized_data_list = [np.array(data).reshape(-1, 1) for data in data_list]

    # Get sequences
    X, y = create_sequences(normalized_data_list, window_size, prediction_offset)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=seed)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    # Create the model
    model = VehicleCountLSTM(input_size = 1, hidden_size = hidden_size, num_layers = num_layers, output_size = 1).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    # Initialize loss files
    train_loss_file = data_dir + f'detection_train_loss_k_{prediction_offset}_seed_{seed}.csv'
    test_loss_file = data_dir + f'detection_test_loss_k_{prediction_offset}_seed_{seed}.csv'
    
    with open(train_loss_file, 'w', newline = '') as f_train, open(test_loss_file, 'w', newline  = '') as f_test:
        train_writer = csv.writer(f_train)
        test_writer = csv.writer(f_test)
        train_writer.writerow(['Epoch', 'Loss'])
        test_writer.writerow(['Epoch', 'Loss'])

        # Calculate initial test loss
        initial_test_loss = calculate_loss(model, criterion, X_test, y_test)
        test_writer.writerow([0, initial_test_loss])
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]    
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

            # Calculate average train loss
            avg_train_loss = train_loss / (len(X_train) / batch_size)
            train_writer.writerow([epoch + 1, avg_train_loss])

            # Calculate test loss
            test_loss = calculate_loss(model, criterion, X_test, y_test)
            test_writer.writerow([epoch + 1, test_loss])
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}')
	
    return model

def save_model(model, window_size, prediction_offset, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, f'lstm_w_{window_size}_o_{prediction_offset}.pth')
    torch.save(model.state_dict(), model_path)

def train_lstm_model(csv_files, window_size, prediction_offset, data_dir, output_dir, seed):
    set_seed(seed)
    data_list = load_vehicle_count_data(csv_files)
    model = train_lstm(data_list, window_size, prediction_offset, data_dir)
    save_model(model, window_size, prediction_offset, output_dir)

if __name__ == '__main__':
    SEED = 0
    data_dir = '../data/'
    output_dir = '../../../models/lstm/'

    window_size = 3
    prediction_offsets = list(range(20)) + list(range(24, 100, 5))
    
    csv_files = get_vehicle_detection_csv_files(data_dir)

    for prediction_offset in prediction_offsets:
        print(f"Training LSTM for offset {prediction_offset}")
        
        train_lstm_model(csv_files, window_size, prediction_offset, data_dir, output_dir, SEED)


# Try exponential/log loss (take the exponential over MSE) (e ** (0.1 * MSE)) or 1 + log (0.1 * MSE)
# Decrease the window size to 2 
