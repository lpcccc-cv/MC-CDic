'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt, sampler):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2,
                                           pin_memory=True)


def create_dataset(dataset_opt, train):
    mode = dataset_opt['mode']
    if mode == 'IXI':
        from data.IXI_dataset import IXI_train as D
    # elif mode == 'fastmri':
        # from data.fastmri_dataset import SliceDataset as D
    elif mode == 'knee':
        from data.knee_dataset import fastmri_train as D
    elif mode == 'brain':
        from data.brain_dataset import brain_train as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt, train)

    return dataset
