# from langchain.agents import tool #old way
from langchain.tools import tool #Feb 2026 recent tool call
import datetime

@tool
def get_system_time(format:str = "%Y-%m-%d %H:%M:%S"):
    """ Return the current date and time in the specified format"""
    current_time = datetime.datetime.now()
    formatted_time  =current_time.strftime(format)
    return formatted_time