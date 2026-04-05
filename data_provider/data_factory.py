from data_provider.classification_loader import ClassificationDataset, classification_collate_fn
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, Dataset_PEMS, \
    Dataset_Solar, Dataset_ECL, Dataset_Traffic
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'PEMS': Dataset_PEMS,
    'ECL': Dataset_ECL,
    'Solar': Dataset_Solar,
    'Traffic': Dataset_Traffic,
}


def data_provider(args, flag):
    normalized_flag = flag.lower()
    collate_fn = None

    if args.task_name == 'classification':
        if normalized_flag not in ['train', 'val', 'test']:
            raise ValueError('Unsupported classification split: {}'.format(flag))
        data_set = ClassificationDataset(
            root_path=args.root_path,
            flag=normalized_flag,
            val_ratio=getattr(args, 'classification_val_ratio', 0.2),
            seed=getattr(args, 'classification_split_seed', 42),
        )
        shuffle_flag = normalized_flag == 'train'
        drop_last = False
        batch_size = args.batch_size
        collate_fn = classification_collate_fn
        print(normalized_flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn)
        return data_set, data_loader

    if args.data not in data_dict:
        raise ValueError('Unsupported dataset: {}'.format(args.data))
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if normalized_flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif normalized_flag == 'pred':
        raise NotImplementedError('Prediction mode requires Dataset_Pred, which is not implemented in this repository.')
    elif normalized_flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=normalized_flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(normalized_flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn)
    return data_set, data_loader
