import time
from loguru import logger


def generate_text_chat(client, *args, **kwargs):
    e = ''
    for _ in range(25):
        try:
            response = client.chat.completions.create(*args, **kwargs)
            time.sleep(0.5)
            if response is None:
                time.sleep(30)
                continue
            return response
        except Exception as e:
            logger.info(e)
            time.sleep(30)
    return None