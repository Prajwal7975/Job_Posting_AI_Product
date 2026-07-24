import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Create a detailed error message containing
    the file name, line number, and original error.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return str(error)

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in script: [{file_name}] "
        f"at line: [{line_number}] "
        f"Error message: [{str(error)}]"
    )


class CustomException(Exception):
    """
    Custom exception class used throughout the project.
    """

    def __init__(self, error: Exception, error_detail: sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self) -> str:
        return self.error_message