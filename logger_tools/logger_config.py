import logging

def get_logger(name):
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # [수정된 부분] 포맷에 파일명, 함수명, 라인 번호를 추가했습니다.
        log_format = '[%(asctime)s] [%(filename)s > %(funcName)s():%(lineno)d] [%(levelname)s] - %(message)s'
        formatter = logging.Formatter(log_format)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger