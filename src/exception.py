import sys
from logger import logging


def get_error_message_details(error, error_detail: sys):
    # Getting info where the exeption occured
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_message = f"Error occured in python script name [{file_name}] line number [{line_no}] error message[{str(error)}]"

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = get_error_message_details(error_message, error_detail)

    def __str__(self):
        return self.error_message
