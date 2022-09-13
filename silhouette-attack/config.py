conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {
        #'dataset_path': "root_path_for_output",
        'dataset_path': "/hd2/heziwen/GaitDatasetA_out/",#"/hd2/heziwen/GaitDatasetB_out/",
        'resolution': '64',
        #'dataset': 'CASIA-B',
        'dataset': 'CASIA-A',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 0,
        #'pid_num': 73,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}
