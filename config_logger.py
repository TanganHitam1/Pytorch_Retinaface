#Import library
import logging
from io import StringIO

def logger_config(log_path):
    logger = logging.getLogger(__name__)
    #Logger Config
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)

    # Create a formatter to define the format of the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')

    # Add the formatter to the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

def logger_summary(variable, logger):
    #Logger Summary
    summary = StringIO()
    variable.summary(print_fn=lambda x: summary.write(x + '\n'))
    logger.info(summary.getvalue())

def check_var(var):
  #Checks if a variable is empty (None, empty string, empty list, etc.)
  return var is None or not var

def empty_var_logger(var_name, var_value):
  """Logs a message if the provided variable is empty."""
  if check_var(var_value):
     logging.warning(f"Variable '{var_name}' is empty.")
  else:
     logging.info(f"Variable '{var_name}' is not empty.")