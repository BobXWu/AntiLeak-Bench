from tqdm import tqdm
from utils.logger import get_logger
from utils.file_utils import get_batch_files, jsonl_generator, read_json

logger = get_logger()


class Dataset:
    def __init__(
        self,
        data_path: str
    ):
        self.data = read_json(data_path)
