import os
import logging

def get_logger(path, log_file_name):
    if not os.path.exists(path):
        os.mkdir(path)

    # 日志整体配置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 日志文件相关配置
    file_handler = logging.FileHandler(os.path.join(path, log_file_name), mode='w')
    file_handler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

    # 日志控制台相关配置
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 日志格式设置
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger