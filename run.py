import subprocess

num_epoch_normal = 1
num_epoch_pretraining = 1
lr_normal = 0.001
lr_pretraining = 0.001
temperature = 0.07
device = 'cpu'

# Define the list of files and their corresponding arguments
file_args = [
    ('trip_resnet_cwt.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_resnet_cwt.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_resnet_cwt.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_resnet_cwt.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_transformer.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_transformer.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_transformert.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_transformer.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_lstm_linear.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_lstm_linear.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_lstm_linear.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_lstm_linear.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_lstm_attention.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_lstm_attention.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_lstm_attention.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_lstm_attention.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_lstm.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_lstm.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_lstm.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_lstm.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_conv1.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_conv1.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_conv1.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_conv1.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_bilstm.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_bilstm.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_bilstm.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_bilstm.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_combined.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('trip_combined.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}')
    ('cont_combined.py', '3', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('cont_combined.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}'),
    ('combined.py', '6', f'{num_epoch_normal}', f'{num_epoch_pretraining}', f'{lr_normal}', f'{lr_pretraining}', f'{temperature}', f'{device}')
]

# Iterate over the list and execute each file with its arguments
for file_arg in file_args:
    filename, *arguments = file_arg
    command = ['python', filename] + arguments
    subprocess.run(command)
    print('****************************************************************')
    print()

