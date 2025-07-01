# mt5_path = "./pretrained_weight/mt5-base"
mt5_path = "t5-small"

# label paths
train_label_paths = {
                    "CSL_News": "/mnt/fast/nobackup/scratch4weeks/ef0036/cslnews/data/train/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train",
                    "BOBSL": "/mnt/fast/nobackup/scratch4weeks/ks0085/datasets/bobsl/bobsl.train.subset.csv"
                    }

dev_label_paths = {
                    "CSL_News": "/mnt/fast/nobackup/scratch4weeks/ef0036/cslnews/data/train/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.dev",
                    "WLASL": "./data/WLASL/labels-2000.dev",
                    "BOBSL": "/mnt/fast/nobackup/scratch4weeks/ks0085/datasets/bobsl/bobsl.dev.csv"
                    }

test_label_paths = {
                    "CSL_News": "/mnt/fast/nobackup/scratch4weeks/ef0036/cslnews/data/train/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.test",
                    "WLASL": "./data/WLASL/labels-2000.test",
                    "BOBSL": "/mnt/fast/nobackup/scratch4weeks/ks0085/datasets/bobsl/bobsl.test.csv"
                    }


# video paths
rgb_dirs = {
            "CSL_News": '/mnt/fast/nobackup/scratch4weeks/ef0036/cslnews/rgb_format',
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format"

            }

# pose paths
pose_dirs = {
            "CSL_News": '/mnt/fast/nobackup/scratch4weeks/ef0036/cslnews/rgb_format',
            "CSL_Daily": '/mnt/fast/nobackup/scratch4weeks/ef0036/csl_daily/pose_format',
            "WLASL": "/mnt/fast/nobackup/scratch4weeks/ef0036/wlasl",
            "BOBSL": "/mnt/fast/nobackup/scratch4weeks/ks0085/datasets/bobsl/mediapipe"
            }