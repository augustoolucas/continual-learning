from avalanche.benchmarks import SplitMNIST
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler


def load_data(dataset, n_experiences=5):
    if dataset == 'SplitMNIST':
        data = SplitMNIST(n_experiences=n_experiences,
                          shuffle=True,
                          dataset_root='data/datasets',
                          return_task_id=True)
    else:
        raise ValueError(f'dataset {dataset} not supported.')

    return data


def get_dataloader(dataset, batch_size, imbalanced_sampler=False):
    """
    Creates a dataloader for the given dataset.

    Args
    ----------
    dataset : Dataset
        The dataset for which the dataloader is going to be created
    batch_size : int
        The batch size
    imbalanced_sampler : bool
        Whether or not to use a data sampler for imbalanced data

    Returns
    ----------
    torch.utils.data.Dataloader
    """

    sampler = ImbalancedDatasetSampler(dataset) if imbalanced_sampler else None
    shuffle = False if sampler is not None else True
    dataloader = DataLoader(dataset=dataset,
                            num_workers=4,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            pin_memory=True,
                            sampler=sampler)

    return dataloader
