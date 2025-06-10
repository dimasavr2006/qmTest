# #!/usr/bin/env python3

# import argparse
# import pickle
# import sys
# import os # Добавлено для работы с путями
# import numpy as np # Добавлено для загрузки npy

# import torch

# sys.path.append('../')
# from train import train


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('dataset_trained')
#     parser.add_argument('basis_set')
#     parser.add_argument('radius')
#     parser.add_argument('grid_interval')
#     parser.add_argument('dim', type=int)
#     parser.add_argument('layer_functional', type=int)
#     parser.add_argument('hidden_HK', type=int)
#     parser.add_argument('layer_HK', type=int)
#     parser.add_argument('operation')
#     parser.add_argument('batch_size', type=int)
#     parser.add_argument('lr', type=float)
#     parser.add_argument('lr_decay', type=float)
#     parser.add_argument('step_size', type=int)
#     parser.add_argument('iteration', type=int)
#     parser.add_argument('setting')
#     parser.add_argument('num_workers', type=int)
#     parser.add_argument('dataset_predict')
#     args = parser.parse_args()
#     dataset_trained = args.dataset_trained
#     basis_set = args.basis_set
#     radius = args.radius
#     grid_interval = args.grid_interval
#     dim = args.dim
#     layer_functional = args.layer_functional
#     hidden_HK = args.hidden_HK
#     layer_HK = args.layer_HK
#     operation = args.operation
#     batch_size = args.batch_size
#     lr = args.lr
#     lr_decay = args.lr_decay
#     step_size = args.step_size
#     iteration = args.iteration
#     setting = args.setting
#     num_workers = args.num_workers
#     dataset_predict = args.dataset_predict

#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')

#     dir_trained = '../dataset/' + dataset_trained + '/'
#     dir_predict = '../dataset/' + dataset_predict + '/'

#     field = '_'.join([basis_set, radius + 'sphere', grid_interval + 'grid/'])
#     dataset_test = train.MyDataset(dir_predict + 'test_' + field)
#     dataloader_test = train.mydataloader(dataset_test, batch_size=batch_size,
#                                          num_workers=num_workers)

#     with open(dir_trained + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
#         orbital_dict = pickle.load(f)
#     N_orbitals = len(orbital_dict)

#     N_output = len(dataset_test[0][-2][0])

#     model = train.QuantumDeepField(device, N_orbitals,
#                                    dim, layer_functional, operation, N_output,
#                                    hidden_HK, layer_HK).to(device)
#     model.load_state_dict(torch.load('../pretrained_model/model--' + setting,
#                                      map_location=device))
#     tester = train.Tester(model)

#     print('Start predicting for', dataset_predict, 'dataset.\n'
#           'using the pretrained model with', dataset_trained, 'dataset.\n'
#           'The prediction result is saved in the output directory.\n'
#           'Wait for a while...')

#     # MAE, prediction = tester.test(dataloader_test, time=True)
#     prediction = tester.predict(dataloader_test, time=True)
#     filename = ('../output/prediction--' + dataset_predict +
#                 '--' + setting + '.txt')
#     tester.save_prediction(prediction, filename)

#     # print('MAE:', MAE)

#     print('The prediction has finished.')
#!/usr/bin/env python3

import argparse
import pickle
import sys
import os # Добавлено для работы с путями
import numpy as np # Добавлено для загрузки npy

import torch

sys.path.append('../')
from train import train # Предполагается, что MyDataset и другие классы/функции train здесь

if __name__ == "__main__":
    # ... (парсер аргументов и получение args как в вашем коде) ...
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_trained')
    parser.add_argument('basis_set')
    parser.add_argument('radius') # Должен быть float
    parser.add_argument('grid_interval') # Должен быть float
    parser.add_argument('dim', type=int)
    parser.add_argument('layer_functional', type=int)
    parser.add_argument('hidden_HK', type=int)
    parser.add_argument('layer_HK', type=int)
    parser.add_argument('operation')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('lr_decay', type=float)
    parser.add_argument('step_size', type=int)
    parser.add_argument('iteration', type=int)
    parser.add_argument('setting')
    parser.add_argument('num_workers', type=int)
    parser.add_argument('dataset_predict')
    args = parser.parse_args()
    dataset_trained = args.dataset_trained
    basis_set = args.basis_set
    radius = str(args.radius) # Преобразуем обратно в строку для field, если нужно, или используем args.radius напрямую как float
    grid_interval = str(args.grid_interval) # Аналогично
    dim = args.dim
    layer_functional = args.layer_functional
    hidden_HK = args.hidden_HK
    layer_HK = args.layer_HK
    operation = args.operation
    batch_size = args.batch_size
    lr = args.lr
    lr_decay = args.lr_decay
    step_size = args.step_size
    iteration = args.iteration
    setting = args.setting
    num_workers = args.num_workers
    dataset_predict = args.dataset_predict


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dir_trained_base = '../dataset/' + dataset_trained + '/' # Базовый каталог тренировочных данных
    dir_predict_base = '../dataset/' + dataset_predict + '/' # Базовый каталог данных для предсказания

    # --- Определение N_output из обработанных ТРЕНИРОВОЧНЫХ данных ---
    # Формируем путь к обработанным тренировочным данным
    # Убедитесь, что radius и grid_interval здесь строки с нужным форматом
    field_suffix_trained = '_'.join([basis_set, radius + 'sphere', grid_interval + 'grid/'])
    
    # Попробуем сначала 'train', потом 'val' подкаталоги для обработанных данных
    processed_trained_data_dir = os.path.join(dir_trained_base, 'train_' + field_suffix_trained)
    if not os.path.isdir(processed_trained_data_dir):
        processed_trained_data_dir = os.path.join(dir_trained_base, 'val_' + field_suffix_trained)
    
    if not os.path.isdir(processed_trained_data_dir):
        print(f"Error: Processed training data directory not found for {dataset_trained}.")
        print(f"Looked for 'train_{field_suffix_trained}' or 'val_{field_suffix_trained}' in {dir_trained_base}")
        sys.exit(1)

    # Найдем первый .npy файл в этой директории
    npy_files = [f for f in os.listdir(processed_trained_data_dir) if f.endswith('.npy')]
    if not npy_files:
        print(f"Error: No .npy files found in processed training directory: {processed_trained_data_dir}")
        sys.exit(1)
    
    sample_trained_item_path = os.path.join(processed_trained_data_dir, npy_files[0])
    try:
        sample_trained_item = np.load(sample_trained_item_path, allow_pickle=True)
        # В тренировочных данных, созданных с property=True, свойства находятся на предпоследнем месте
        # sample_trained_item[-2] это property_values (формат np.array([[v1, v2, ...]]))
        N_output = len(sample_trained_item[-2][0]) 
    except Exception as e:
        print(f"Error loading or parsing sample from processed trained data ({sample_trained_item_path}): {e}")
        sys.exit(1)
    
    print(f"Determined N_output = {N_output} from dataset_trained: {dataset_trained}")

    # --- Загрузка orbital_dict ---
    orbital_dict_path = os.path.join(dir_trained_base, 'orbitaldict_' + basis_set + '.pickle')
    with open(orbital_dict_path, 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals = len(orbital_dict)

    # --- Подготовка данных для ПРЕДСКАЗАНИЯ ---
    # Эти данные должны были быть созданы predict/preprocess.py с property=False
    field_suffix_predict = '_'.join([basis_set, radius + 'sphere', grid_interval + 'grid/'])
    dir_processed_predict = os.path.join(dir_predict_base, 'test_' + field_suffix_predict)

    if not os.path.isdir(dir_processed_predict):
        print(f"Error: Processed prediction data directory not found: {dir_processed_predict}")
        print(f"Please run predict/preprocess.py for dataset '{dataset_predict}' first.")
        sys.exit(1)
        
    dataset_test = train.MyDataset(dir_processed_predict) # MyDataset загружает данные без свойств
    if len(dataset_test) == 0:
        print(f"Error: No data found in processed prediction directory: {dir_processed_predict}")
        print(f"This might indicate an issue with predict/preprocess.py or the input file for '{dataset_predict}'.")
        sys.exit(1)

    dataloader_test = train.mydataloader(dataset_test, batch_size=batch_size,
                                         num_workers=num_workers)

    # --- Инициализация и загрузка модели ---
    # N_output теперь определен корректно
    model = train.QuantumDeepField(device, N_orbitals,
                                   dim, layer_functional, operation, N_output,
                                   hidden_HK, layer_HK).to(device)
    
    model_path = '../pretrained_model/model--' + setting
    if not os.path.exists(model_path):
        print(f"Error: Pretrained model file not found: {model_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    tester = train.Tester(model)

    print(f'\nStart predicting for {dataset_predict} dataset.\n'
          f'using the pretrained model from {setting} (trained on {dataset_trained}).\n'
          f'The prediction result will be saved in the output directory.')

    prediction = tester.predict(dataloader_test, time=True)
    
    output_dir = '../output/'
    os.makedirs(output_dir, exist_ok=True) # Создаем директорию, если ее нет
    filename = os.path.join(output_dir, f'prediction--{dataset_predict}--{setting}.txt')
    
    tester.save_prediction(prediction, filename)

    print(f'\nThe prediction has finished. Results saved to: {filename}')
